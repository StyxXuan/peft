import tvm
from tvm import te
import torch
import numpy as np
from tvm.contrib import dlpack


class IndexedMatMul_A(torch.autograd.Function):

    # 用于存储已编译的函数
    function_dict = {}

    @staticmethod
    def compile_tvm_func(dtype="float32", device="cuda",):
        """
        编译 TVM 函数（通用编译，不受 n, d, k, r 维度影响）
        """
        n = te.var("n")
        d = te.var("d")
        k = te.var("k")
        r = te.var("r")
        
        X_tvm = te.placeholder((n, d), name="X", dtype=dtype)
        M_tvm = te.placeholder((n, k), name="M", dtype="int32")
        A_tvm = te.placeholder((r, d), name="A", dtype=dtype)

        k_axis = te.reduce_axis((0, d), name="k_axis")
        Y_tvm = te.compute(
            (n, k),
            lambda i, j: te.sum(X_tvm[i, k_axis] * A_tvm[M_tvm[i, j], k_axis], axis=k_axis),
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
    
        func = tvm.build(s, [X_tvm, M_tvm, A_tvm, Y_tvm], target=device)
        return func

    @staticmethod
    def load_compiled_func(dtype="float32", device="cuda", path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/peft/src/peft/tuners/srmole/tvm_func_A"):
        """
        从文件加载已编译的 TVM 函数
        """
        key = f"{dtype}_{device}"
        if key not in IndexedMatMul_A.function_dict:
            # 加载编译好的函数
            print(f"加载已编译的函数：{path}_{key}.so")
            loaded_func = tvm.runtime.load_module(f"{path}_{key}.so")
            # 转换为 PyTorch 可调用函数
            loaded_func_pytorch = dlpack.to_pytorch_func(loaded_func)
            IndexedMatMul_A.function_dict[key] = loaded_func_pytorch
        return IndexedMatMul_A.function_dict[key]

    @staticmethod
    def _get_function(dtype: str, device: str):
        """
        优先从缓存获取函数，如果没有则尝试从文件加载，
        如果文件不存在则重新编译并保存
        """
        key = f"{dtype}_{device}"
        if key not in IndexedMatMul_A.function_dict:
            try:
                # 尝试从文件加载
                print("从文件中加载函数...")
                func = IndexedMatMul_A.load_compiled_func(dtype=dtype, device=device)
                print("从文件加载函数到缓存成功。")
                return func
            except:
                # 如果加载失败，则重新编译并保存
                print("文件中加载函数失败，重新编译函数并保存...")
                indexed_matmul = IndexedMatMul_A.compile_tvm_func(dtype=dtype, device=device)
                indexed_matmul_pytorch = dlpack.to_pytorch_func(indexed_matmul)
                IndexedMatMul_A.function_dict[key] = indexed_matmul_pytorch
                # 保存编译后的函数
                indexed_matmul.export_library(f"/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/peft/src/peft/tuners/srmole/tvm_func_A_{dtype}_{device}.so")
                print("编译保存函数成功。")
                
        return IndexedMatMul_A.function_dict[key]
    
    @staticmethod
    def _prepare_tensors(t):
        assert t.is_contiguous()
        return t

    @staticmethod
    def _indexed_matmul(X_torch: torch.Tensor, M_torch: torch.Tensor, A_torch: torch.Tensor):
        X_torch = IndexedMatMul_A._prepare_tensors(X_torch)
        A_torch = IndexedMatMul_A._prepare_tensors(A_torch)
        M_torch = IndexedMatMul_A._prepare_tensors(M_torch)
        dtype = str(X_torch.dtype).split('.')[1]
        device = X_torch.device.type

        # 获取实际的尺寸
        n, d = X_torch.shape
        _, k = M_torch.shape
        r, _ = A_torch.shape

        Y = torch.empty((n, k), dtype=X_torch.dtype, device=X_torch.device)
        func = IndexedMatMul_A._get_function(dtype, device)

        # 调用编译后的函数
        func(X_torch, M_torch, A_torch, Y)
        return Y

    @staticmethod
    def forward(ctx, X_torch, M_torch, A_torch):        
        ctx.save_for_backward(X_torch, M_torch, A_torch)
        Y_torch  = IndexedMatMul_A._indexed_matmul(X_torch=X_torch, M_torch=M_torch, A_torch=A_torch)
        return Y_torch
    
    @staticmethod
    def backward(ctx, grad_output):
        '''
        基于torch实现
        '''
        X_torch, M_torch, A_torch = ctx.saved_tensors
        dtype = str(grad_output.dtype).split('.')[1]
        device = grad_output.device.type

        # 计算 dL/dX
        grad_X = torch.einsum('nk,nkd->nd', grad_output, A_torch[M_torch.long()])

        # 计算 dL/dA
        grad_A = torch.zeros_like(A_torch)
        n, k = M_torch.shape
        d = X_torch.shape[1]
        M_flattened = M_torch.view(-1)  # (n*k)
        src_flattened = (grad_output.view(n, k, 1) * X_torch.unsqueeze(1)).view(-1, d)  # (n*k, d)
        grad_A.index_add_(0, M_flattened.long(), src_flattened)

        return grad_X, None, grad_A


import time
def test_indexed_matmul():
    test_cases = [
        (1024, 512, 10, 256),
        # 可以根据需要添加更多测试用例
    ]
    device = torch.device("cuda")
    
    for n, d, k, r in test_cases: 
        print("-" * 50)
        print(f"测试维度: n={n}, d={d}, k={k}, r={r}")
        
        # 数据生成
        torch.manual_seed(42)
        X = torch.rand((n, d), device=device, requires_grad=True).to(torch.bfloat16)
        M = torch.randint(0, r, (n, k), device=device, dtype=torch.int32)
        A = torch.rand((r, d), device=device, requires_grad=True).to(torch.bfloat16)
    
        # PyTorch 实现
        def pytorch_impl(X, M, A):
            A_sub = A[M.long(), :]  # 形状为 (n, k, d)
            Y_torch = torch.einsum('nd,nkd->nk', X, A_sub)
            return Y_torch

        start_time = time.time()
        Y_pytorch = pytorch_impl(X, M, A)
        pytorch_time = time.time() - start_time
        print(f"PyTorch 前向计算时间：{pytorch_time:.5f} 秒")

        # TVM + PyTorch 实现
        start_time = time.time()
        Y_tvm = IndexedMatMul_A.apply(X, M, A)
        tvm_time = time.time() - start_time
        print(f"TVM 前向计算时间：{tvm_time:.5f} 秒")


        # 验证结果
        torch.testing.assert_close(Y_pytorch, Y_tvm, rtol=1e-5, atol=1e-5)
        print("验证成功：TVM 和 PyTorch Einsum 计算结果一致！")
        print(f"TVM 速度相比 PyTorch Einsum 快了 {100 * (pytorch_time - tvm_time) / pytorch_time:.2f}%")

        # 测试反向传播
        loss = Y_tvm.sum()
        t1 = time.time()
        loss.backward()
        t2 = time.time()
        backward_time = t2 - t1
        print(f"梯度计算完成，用时{backward_time}")
        print(f"X 梯度大小: {X.grad}")
        print(f"A 梯度大小: {A.grad}")

if __name__ == "__main__":
    test_indexed_matmul()
