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


class SRMoLEQuantLinear(torch.nn.Module, SRMoLELayer):
    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        activate_r: int = 0,
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
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
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

            lora_A_weight = self.lora_A[active_adapter].weight  # 形状为 (r, d)
            lora_B_weight = self.lora_B[active_adapter].weight  # 形状为 (d, r)

            router = self.router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            router_output = router(x)  # 形状为 (b, s, r)
            router_output = F.softmax(router_output, dim=2)  # 在 r 维度上应用 softmax

            if self.training:
                router_output_flat = router_output.view(-1, router_output.size(-1))
                p = self.soft_topk(router_output_flat)
                p = p.sum(dim=2)
                p = p.view(router_output.size(0), router_output.size(1), -1) # shape (b,s,r)
                
                mid_output = torch.einsum("bsd,rd->bsr", (dropout(x), lora_A_weight))
                lora_outpout = torch.einsum("bsr,dr->bsrd", (mid_output, lora_B_weight))
                lora_outpout = torch.einsum("bsrd,bsr->bsd", (mid_output, p))
                
            else:
                # 选择 top activate_r 参数基于 softmax 得分
                _, indices = torch.topk(router_output, self.activate_r[active_adapter], dim=2)  # indices 形状为 (b, s, k)

                # 使用 gather 获取 selected_lora_A_weight 和 selected_lora_B_weight
                selected_lora_A_weight = lora_A_weight[indices]  # 形状为 (b, s, k, d)
                
                # 获取 selected_lora_B_weight，变为 (b, s, k, d)
                selected_lora_B_weight = lora_B_weight[:, indices]  # 形状为 (d, b, s, k)
                selected_lora_B_weight = selected_lora_B_weight.permute(1, 2, 0, 3)  # 变为 (b, s, d, k)

                # 计算 selected_lora_A_output
                selected_lora_A_output = torch.einsum("bsd,bskd->bsk", (dropout(x), selected_lora_A_weight))  # (b, s, k)

                # 计算 selected_lora_B_output
                lora_outpout = torch.einsum("bsk,bsdk->bsd", (selected_lora_A_output, selected_lora_B_weight))  # (b, s, d)
                if requires_conversion:
                    lora_outpout = lora_outpout.to(expected_dtype)

                result = result + lora_outpout * scaling
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep
