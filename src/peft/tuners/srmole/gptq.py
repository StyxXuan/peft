# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
from .layer import SRMoLELayer
from .topk import TopK_custom
from .idx_matmul_A import IndexedMatMul_A
from .idx_matmul_B import IndexedMatMul_B


class SRMoLEQuantLinear(torch.nn.Module, SRMoLELayer):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        activate_r: int = 0,
        epsilon_greedy: bool = False,
        rank_partition: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        SRMoLELayer.__init__(self, base_layer)
        # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
        # for backwards compatibility
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, activate_r, epsilon_greedy, rank_partition, lora_alpha, lora_dropout, init_lora_weights)
        self.soft_topk = TopK_custom(activate_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # note: no check for self.merged because merging is not supported (yet)
        result = self.base_layer(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_router = self.lora_router[active_adapter]
            activate_r = self.activate_r[active_adapter]
            epsilon_greedy = self.epsilon_greedy[active_adapter]
            rank_partition = self.rank_partition[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # 获取路由结果
            x_dropped = dropout(x)  # [batch_size, seq_len,in_features]
            logits = lora_router(x)
            top_k_logits, indices = logits.topk(activate_r, dim=-1)
            zeros = torch.full_like(logits, float('-inf'))
            sparse_logits = zeros.scatter(-1, indices, top_k_logits)
            gating_output = F.softmax(sparse_logits, dim=-1)
            # gating_output, indices = router(x)  # gating_output: [batch_size, seq_len, r], indices: [batch_size, seq_len, k]

            # flaten
            x_dropped = x_dropped.view(-1, x_dropped.size(-1)) # [batch_size*seq_len, in_features]
            gating_output = gating_output.view(-1, gating_output.size(-1)) # [batch_size*seq_len, r]
            indices = indices.view(-1, indices.size(-1)).to(torch.int32) # [batch_size*seq_len, k]

            # 使用 IndexedMatMul 进行计算
            # 1. 准备权重
            A = lora_A.weight  # [r, in_features]
            B = lora_B.weight  # [out_features, r]

            # 2. 第一步矩阵乘法：x @ A[indices]
            intermediate = IndexedMatMul_A.apply(x_dropped, indices, A)  # [batch_size*seq_len, k]

            # 3. 应用 gating scores
            selected_gates = torch.gather(gating_output, 1, indices.to(torch.int64))  # [batch_size*seq_len, k]
            gated_intermediate = intermediate * selected_gates  # [batch_size*seq_len, k]
            gated_intermediate = gated_intermediate.to(x_dropped.dtype)

            # 4. 第二步矩阵乘法：(x @ A[indices]) @ B[:, indices]
            output = IndexedMatMul_B.apply(gated_intermediate, indices.to(torch.int32), B)  # [batch_size*seq_len, out_features]

            # unflatten
            output = output.view(x.size(0), x.size(1), output.size(-1))

            if requires_conversion:
                lora_outpout = lora_outpout.to(expected_dtype)

            result = result + lora_outpout * scaling
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep
