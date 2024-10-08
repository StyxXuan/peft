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

import warnings
from typing import Any, List, Optional
import torch.nn.functional as F
import math
import packaging
import torch
import transformers
from torch import nn

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
from .topk import TopK_custom

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config


class SRMoLELayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "activate_r")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.activate_r = {}
        self.router = nn.ParameterDict({})


    def update_layer(self, adapter_name, r, activate_r, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.activate_r[adapter_name] = activate_r

        # Actual trainable parameters
        # Right singular vectors
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        # The current rank
        self.router[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.scaling[adapter_name] = lora_alpha / activate_r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

class SRMoLELinear(nn.Module, SRMoLELayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        activate_r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        SRMoLELayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, activate_r, lora_alpha, lora_dropout, init_lora_weights)
        self.soft_topk = TopK_custom(activate_r)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                
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
                    selected_lora_A_weight = torch.einsum("rd,bsr->bsrd", lora_A_weight, p)
                    selected_lora_B_weight = torch.einsum("dr,bsr->bsdr", lora_B_weight, p)
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
                selected_lora_B_output = torch.einsum("bsk,bsdk->bsd", (selected_lora_A_output, selected_lora_B_weight))  # (b, s, d)
                # 更新结果
                result = result + selected_lora_B_output * scaling
                
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep

