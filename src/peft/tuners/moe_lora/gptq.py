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

            lora_A_weight = self.lora_A[active_adapter].weight
            lora_B_weight = self.lora_B[active_adapter].weight

            router = self.router[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            router_output= router(x)
            router_output = F.softmax(router_output, dim=0)  # Apply softmax across the rows
        
            # Select top activate_r parameters based on softmax scores
            _, indices = torch.topk(router_output, self.activate_r[active_adapter], dim=0)

            selected_lora_A_weight = lora_A_weight[indices.squeeze(0)]  # Select active rows
            selected_lora_B_weight = lora_B_weight[:, indices.squeeze(0)]  # Select corresponding columns

            selected_lora_A_output = dropout(x) @ selected_lora_A_weight.T
            selected_lora_B_output = selected_lora_A_output @ selected_lora_B_weight.T
            if requires_conversion:
                selected_lora_B_output = selected_lora_B_output.to(expected_dtype)

            result = result + selected_lora_B_output * scaling

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep
