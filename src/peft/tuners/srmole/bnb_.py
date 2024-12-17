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

from typing import Any

import torch
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .layer import SRMoLELayer
from .layer import TopK_custom


if is_bnb_available():

    class SRMoLELinear8bitLt(torch.nn.Module, SRMoLELayer):
        # Low-rank matrix for SVD-based adaptation
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
            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False

            self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, activate_r, lora_alpha, lora_dropout, init_lora_weights)
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

                gating_output, indices = router(x)  # 形状为 (b, s, r)

                lora_outpout = torch.zeros_like(x)

                # Reshape inputs for batch processing
                flat_x = dropout(x).view(-1, x.size(-1))
                flat_gating_output = gating_output.view(-1, gating_output.size(-1))
                for i in range(self.r[active_adapter]):
                    # Create a mask for the inputs where the current expert is in top-k
                    expert_mask = (indices == i).any(dim=-1)
                    flat_mask = expert_mask.view(-1)

                    if flat_mask.any():
                        expert_input = flat_x[flat_mask]
                        expert_output = expert_input @ lora_A_weight[i].unsqueeze(0).t()
                        expert_output = expert_output @ lora_B_weight[:,i].unsqueeze(0)
                        # Extract and apply gating scores
                        gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                        weighted_output = expert_output * gating_scores
                        # Update final output additively by indexing and adding
                        lora_outpout[expert_mask] += weighted_output.squeeze(1)
                # 更新结果
                if requires_conversion:
                    lora_outpout = lora_outpout.to(expected_dtype)

                result = result + lora_outpout * scaling

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "srmole." + rep


if is_bnb_4bit_available():

    class SRMoLELinear4bitLt(torch.nn.Module, SRMoLELayer):
        # Low-rank matrix for SVD-based adaptation
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
            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False

            self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, activate_r, lora_alpha, lora_dropout, init_lora_weights)
            self.soft_topk = TopK_custom(activate_r)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # note: no check for self.merged because merging is not supported (yet)
            print("4-bit forward")
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

                gating_output, indices = router(x)  # 形状为 (b, s, r)

                lora_outpout = torch.zeros_like(x)

                # Reshape inputs for batch processing
                flat_x = dropout(x).view(-1, x.size(-1))
                flat_gating_output = gating_output.view(-1, gating_output.size(-1))
                for i in range(self.r[active_adapter]):
                    # Create a mask for the inputs where the current expert is in top-k
                    expert_mask = (indices == i).any(dim=-1)
                    flat_mask = expert_mask.view(-1)

                    if flat_mask.any():
                        expert_input = flat_x[flat_mask]
                        expert_output = expert_input @ lora_A_weight[i].unsqueeze(0).t()
                        expert_output = expert_output @ lora_B_weight[:,i].unsqueeze(0)
                        # Extract and apply gating scores
                        gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                        weighted_output = expert_output * gating_scores
                        # Update final output additively by indexing and adding
                        lora_outpout[expert_mask] += weighted_output.squeeze(1)
                if requires_conversion:
                    lora_outpout = lora_outpout.to(expected_dtype)

                result = result + lora_outpout * scaling

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "srmole." + rep