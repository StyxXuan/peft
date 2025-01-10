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
from torch import nn
import torch.nn.functional as F
from .layer import MoELoRALayer


class MoELoRAQuantLinear(torch.nn.Module, MoELoRALayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        expert_num: int = 0,
        routing_strategy: str = "top-1",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        MoELoRALayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, expert_num, lora_alpha, lora_dropout, init_lora_weights)
        self.routing_strategy = routing_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # note: no check for self.merged because merging is not supported (yet)
        result = self.base_layer(x)
        B, S, in_feat = x.shape
        N = B * S
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

            lora_A = self.lora_A[active_adapter]            # [expert_num, in_features, r]
            lora_B = self.lora_B[active_adapter]            # [expert_num, r, out_features]
            lora_router = self.lora_router[active_adapter]  # Linear(in_features -> expert_num)
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            expert_num = self.expert_num[active_adapter]
            r = self.r[active_adapter]

            # 1. 对输入做 drop out
            #    x_dropped: [B, S, in_features]
            x_dropped = dropout(x)

            # 2. 路由器打分: [B, S, expert_num]
            logits = lora_router(x_dropped)

            if self.routing_strategy == "top-1":
                # ================== Top-1 (One-hot) ==================
                # 先对 expert 维度做 softmax，得到概率
                if self.training:
                    gating_probs = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
                    # gating_probs: [B, S, expert_num], 对每个 token 在所有 expert 上的概率分布

                    mid = torch.einsum("bsd,edr->bser", x_dropped, lora_A)
                    mid = torch.einsum("bser,erd->bsed", mid, lora_B)
                    lora_outpout = torch.einsum("bsed,bse->bsd", mid, gating_probs)
                else:
                    top1_vals, top1_idx = torch.max(logits, dim=-1)  # [B, S]

                    # flatten
                    flat_x = x.view(N, in_feat)
                    flat_idx = top1_idx.view(N)  # [N]

                    # 做一个增量输出
                    inc_out = x.new_zeros((N, self.out_features))

                    # 遍历实际出现的 expert (或直接 range(expert_num))
                    experts_unique = flat_idx.unique()
                    for e_id in experts_unique:
                        mask = (flat_idx == e_id)
                        if not mask.any():
                            continue
                        # 取出 token
                        x_sel = flat_x[mask]  # [N_e, in_features]
                        # 取出 LoRA
                        W_A = self.lora_A[e_id]  # [in_features, r]
                        W_B = self.lora_B[e_id]  # [r, out_features]
                        local_out = x_sel @ W_A
                        local_out = local_out @ W_B
                        local_out = local_out * top1_vals.view(-1, 1)[mask]
                        inc_out[mask] = local_out

                    # reshape
                    lora_outpout = inc_out.view(B, S, self.out_features)
            elif self.routing_strategy == "top-2":
                # ================== Top-2 路由 ==================
                # 先对 expert 维度做 softmax 得到概率
                probs = F.softmax(logits, dim=-1, dtype=torch.float)  # [B, S, expert_num]
                # 用 topk 找到前2个专家 (k=2)
                routing_weights, selected_experts = torch.topk(probs, k=2, dim=-1)  
                # [B, S, 2], [B, S, 2]

                # 再次做归一化(只在 top-2 上)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                # cast 回原始 dtype
                routing_weights = routing_weights.to(x.dtype)

                # flatten
                flat_x = x.view(N, in_feat)    # [N, in_features]
                flat_experts = selected_experts.view(N, 2)    # [N, 2]
                flat_weights = routing_weights.view(N, 2)     # [N, 2]

                # 准备最终增量张量
                inc_out = torch.zeros(
                    (N, self.out_features),
                    dtype=x.dtype,
                    device=x.device
                )

                # One-hot 编码 selected_experts => [N, 2, expert_num], 再置换到 [expert_num, N, 2]
                # 也可自己写逻辑手动找 expert id 相等位置，这里和您提供的思路一致
                expert_mask = F.one_hot(flat_experts, num_classes=self.expert_num)  # [N, 2, expert_num]
                expert_mask = expert_mask.permute(2, 0, 1)  # => [expert_num, N, 2]

                # 遍历全部 expert，收集 token
                for e_id in range(self.expert_num):
                    # expert_mask[e_id] 形状 [N, 2]，有 2 列表示第 0 / 1 个专家
                    # torch.where 可以一次性得到 (idx, col)
                    idx, col = torch.where(expert_mask[e_id])  # idx: token 的位置, col: 0 or 1

                    if idx.numel() == 0:
                        continue

                    # 取出 x_sel, w_sel
                    x_sel = flat_x[idx]  # [N_e, in_features]
                    w_sel = flat_weights[idx, col]  # [N_e]
                    w_sel = w_sel.unsqueeze(-1)     # => [N_e, 1]

                    # 取出该 expert 的 LoRA A, B
                    W_A = self.lora_A[e_id]  # [in_features, r]
                    W_B = self.lora_B[e_id]  # [r, out_features]

                    # 一次性计算
                    local_out = x_sel @ W_A  # => [N_e, r]
                    local_out = local_out @ W_B  # => [N_e, out_features]
                    local_out = local_out * w_sel

                    # 用 index_add_ 加回到 inc_out
                    inc_out.index_add_(0, idx, local_out)

                # reshape 并加到 result
                lora_outpout = inc_out.view(B, S, self.out_features)
            else: # soft moe
                gating_probs = F.softmax(logits, dim=-1, dtype=torch.float)  # [B, S, expert_num]
                mid = torch.einsum("bsd,edr->bser", x_dropped, lora_A)
                mid = torch.einsum("bser,erd->bsed", mid, lora_B)
                lora_outpout = torch.einsum("bsed,bse->bsd", mid, gating_probs)
              

            if requires_conversion:
                lora_outpout = lora_outpout.to(expected_dtype)

            result = result + lora_outpout * scaling
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moelora." + rep
