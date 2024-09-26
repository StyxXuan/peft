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

                router_output = router(x)  # 形状为 (b, s, r)
                router_output = F.softmax(router_output, dim=2)  # 在 r 维度上应用 softmax

                # 选择 top activate_r 参数基于 softmax 得分
                _, indices = torch.topk(router_output, self.activate_r[active_adapter], dim=2)  # indices 形状为 (b, s, k)

                # 使用 gather 获取 selected_lora_A_weight 和 selected_lora_B_weight
                selected_lora_A_weight = lora_A_weight[indices]  # 形状为 (b, s, k, d)
                
                # 获取 selected_lora_B_weight，变为 (b, s, k, d)
                selected_lora_B_weight = lora_B_weight[:, indices]  # 形状为 (d, b, s, k)
                selected_lora_B_weight = selected_lora_B_weight.permute(1, 2, 0, 3)  # 变为 (b, s, d, k)

                # print(x.shape)  # 输入 x 的形状
                # print(selected_lora_A_weight.shape)  # 选择后的 lora_A_weight 的形状 (b, s, k, d)
                # print(selected_lora_B_weight.shape)  # 选择后的 lora_B_weight 的形状 (b, s, k, d)

                # 计算 selected_lora_A_output
                selected_lora_A_output = torch.einsum("bsd,bskd->bsk", (dropout(x), selected_lora_A_weight))  # (b, s, k)

                # 计算 selected_lora_B_output
                selected_lora_B_output = torch.einsum("bsk,bsdk->bsd", (selected_lora_A_output, selected_lora_B_weight))  # (b, s, d)
                if requires_conversion:
                    selected_lora_B_output = selected_lora_B_output.to(expected_dtype)

                result = result + selected_lora_B_output * scaling

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

                router_output = router(x)  # 形状为 (b, s, r)
                router_output = F.softmax(router_output, dim=2)  # 在 r 维度上应用 softmax

                # 选择 top activate_r 参数基于 softmax 得分
                _, indices = torch.topk(router_output, self.activate_r[active_adapter], dim=2)  # indices 形状为 (b, s, k)

                # 使用 gather 获取 selected_lora_A_weight 和 selected_lora_B_weight
                selected_lora_A_weight = lora_A_weight[indices]  # 形状为 (b, s, k, d)
                
                # 获取 selected_lora_B_weight，变为 (b, s, k, d)
                selected_lora_B_weight = lora_B_weight.t()[indices].permute(0, 2, 1)  # 形状为 (b, s, k, d)

                print(x.shape)  # 输入 x 的形状
                print(selected_lora_A_weight.shape)  # 选择后的 lora_A_weight 的形状 (b, s, k, d)
                print(selected_lora_B_weight.shape)  # 选择后的 lora_B_weight 的形状 (b, s, k, d)

                # 计算 selected_lora_A_output
                selected_lora_A_output = torch.einsum("bsd,rd->bsr", (dropout(x), selected_lora_A_weight))  # (b, s, k)

                # 计算 selected_lora_B_output
                selected_lora_B_output = torch.einsum("bsr,dk->bsd", (selected_lora_A_output, selected_lora_B_weight))  # (b, s, d)
                if requires_conversion:
                    selected_lora_B_output = selected_lora_B_output.to(expected_dtype)

                result = result + selected_lora_B_output * scaling

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "srmole." + rep
