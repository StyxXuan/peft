import tvm
from tvm import te
import torch
import numpy as np
from tvm.contrib import dlpack

class IndexedMatMul_B(torch.autograd.Function):
    function_dict = {}

    @staticmethod
    def compile_tvm_func(dtype="float32", device="cuda"):
        """
        编译 TVM 函数
        假设:
        X: (n, k)
        M: (n, k)
        B: (d, r)

        计算:
        Y[i, dd] = Σ_j X[i,j]*B[dd,M[i,j]]
        最终 Y 形状为 (n, d)
        """
        n = te.var("n")
        k = te.var("k")
        d = te.var("d")
        r = te.var("r")

        X_tvm = te.placeholder((n, k), name="X", dtype=dtype)
        M_tvm = te.placeholder((n, k), name="M", dtype="int32")
        B_tvm = te.placeholder((d, r), name="B", dtype=dtype)

        j_axis = te.reduce_axis((0, k), name="j")
        Y_tvm = te.compute(
            (n, d),
            lambda i, dd: te.sum(X_tvm[i, j_axis] * B_tvm[dd, M_tvm[i, j_axis]], axis=j_axis),
            name="Y"
        )

        s = te.create_schedule(Y_tvm.op)
        bx, tx = s[Y_tvm].split(Y_tvm.op.axis[0], factor=16)
        by, ty = s[Y_tvm].split(Y_tvm.op.axis[1], factor=16)
        s[Y_tvm].reorder(bx, by, tx, ty)
        s[Y_tvm].bind(bx, te.thread_axis("blockIdx.x"))
        s[Y_tvm].bind(by, te.thread_axis("blockIdx.y"))
        s[Y_tvm].bind(tx, te.thread_axis("threadIdx.x"))
        s[Y_tvm].bind(ty, te.thread_axis("threadIdx.y"))
        
        target_str = "cuda -arch=sm_90"
        func = tvm.build(s, [X_tvm, M_tvm, B_tvm, Y_tvm], target=target_str)
        return func

    @staticmethod
    def load_compiled_func(dtype="float32", device="cuda", path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/peft/src/peft/tuners/srmole/tvm_func_B"):
        key = f"{dtype}_{device}"
        if key not in IndexedMatMul_B.function_dict:
            loaded_func = tvm.runtime.load_module(f"{path}_{key}.so")
            loaded_func_pytorch = dlpack.to_pytorch_func(loaded_func)
            IndexedMatMul_B.function_dict[key] = loaded_func_pytorch
        return IndexedMatMul_B.function_dict[key]

    @staticmethod
    def _get_function(dtype: str, device: str):
        key = f"{dtype}_{device}"
        if key not in IndexedMatMul_B.function_dict:
            try:
                func = IndexedMatMul_B.load_compiled_func(dtype=dtype, device=device)
                return func
            except:
                func_tvm = IndexedMatMul_B.compile_tvm_func(dtype=dtype, device=device)
                func_pytorch = dlpack.to_pytorch_func(func_tvm)
                IndexedMatMul_B.function_dict[key] = func_pytorch
                func_tvm.export_library(f"/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/peft/src/peft/tuners/srmole/tvm_func_B_{dtype}_{device}.so")
        return IndexedMatMul_B.function_dict[key]

    @staticmethod
    def forward(ctx, X, M, B):
        ctx.save_for_backward(X, M, B)
        dtype = str(X.dtype).split('.')[-1]
        device = X.device.type

        n, k = X.shape
        n_, k_ = M.shape
        d, r = B.shape
        assert n == n_ and k == k_, "X和M的shape不匹配"

        Y = torch.empty((n, d), dtype=X.dtype, device=X.device)
        func = IndexedMatMul_B._get_function(dtype, device)
        func(X, M.int(), B, Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向计算梯度:
        dX[i,j] = Σ_dd grad_out[i,dd]*B[dd,M[i,j]]

        dB[dd,col] = Σ_{i,j|M[i,j]=col} grad_out[i,dd]*X[i,j]
        """
        X, M, B = ctx.saved_tensors
        n, k = X.shape
        d, r = B.shape

        # dX
        # 构建 B_sub: (n,k,d)
        # B_sub[i,j,dd] = B[dd,M[i,j]]
        B_sub = B[:, M.long()] # (d,n,k)
        B_sub = B_sub.permute(1,2,0) # (n,k,d)
        grad_X = torch.einsum('nd,nkd->nk', grad_output, B_sub)

        # dB
        # 我们需要对B在col维上加, B是(d,r)
        # 构造src_flattened: (n*k, d)
        # grad_out:(n,d), X:(n,k) => (n,k,d) => flatten成(n*k,d)
        src_flattened = (grad_output.unsqueeze(1)*X.unsqueeze(2)).view(-1, d) # (n*k,d)

        grad_B = torch.zeros_like(B)
        M_flat = M.view(-1)
        # index_add_对列索引加，每次加 (d, n*k)中的一列
        grad_B.index_add_(1, M_flat.long(), src_flattened.T)

        return grad_X, None, grad_B


# 测试代码
def test_indexed_matmul_final():
    device = torch.device("cuda")
    n, k, d, r = 1024, 10, 512, 256
    torch.manual_seed(42)
    X = torch.rand((n, k), device=device, dtype=torch.bfloat16, requires_grad=True)
    M = torch.randint(0, r, (n, k), device=device, dtype=torch.int32)
    B = torch.rand((d, r), device=device, dtype=torch.bfloat16, requires_grad=True)

    # 参考实现:
    # 对每行 i:
    #   B_sub = B[:, M[i,:]] (d,k)
    #   Y[i,:] = X[i,:](1,k) @ B_sub^T(k,d) = (1,d)
    Y_ref = torch.empty((n, d), device=device, dtype=torch.bfloat16)
    for i in range(n):
        B_sub = B[:, M[i].long()]  # (d,k)
        Y_ref[i] = X[i].unsqueeze(0).mm(B_sub.transpose(0,1))

    Y_tvm = IndexedMatMul_B.apply(X, M, B)
    torch.testing.assert_close(Y_ref, Y_tvm, atol=1e-2, rtol=1e-2)
    print("前向计算验证成功")

    loss = Y_tvm.sum()
    loss.backward()

    print("反向传播完成")
    print("X.grad:", X.grad)
    print("B.grad:", B.grad)

if __name__ == '__main__':
    test_indexed_matmul_final()