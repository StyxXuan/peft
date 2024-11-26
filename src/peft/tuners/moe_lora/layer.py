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


if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config


class MoELoRALayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "expert_num")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.expert_num = {}
        self.router = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

    def update_layer(self, adapter_name, r, expert_num, lora_alpha, lora_dropout, init_lora_weights):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.expert_num[adapter_name] = expert_num
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # Actual trainable parameters
        # Right singular vectors
        self.lora_A[adapter_name] = nn.Parameter(
            torch.empty(
                expert_num,
                r,
                self.in_features,
            )
        )

        self.lora_B[adapter_name] = nn.Parameter(
            torch.empty(
                expert_num,
                self.out_features,
                r,
            )
        )

        # The current rank
        self.router[adapter_name] = nn.Linear(self.in_features, expert_num, bias=False)
        self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name])

class MoELoRALinear(nn.Module, MoELoRALayer):
    # SVD-based adaptation by a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        expert_num: int = 0,
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
                
                lora_A_weight = self.lora_A[active_adapter]  # 形状为 (n, r, d)
                lora_B_weight = self.lora_B[active_adapter]  # 形状为 (n, d, r)

                router = self.router[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                router_logits = router(x)  # 形状为 (b, s, n)
                if self.training:
                    # 训练阶段，使用 Gumbel-Softmax，hard=False，使用 soft routing
                    router_probs = F.gumbel_softmax(router_logits, tau=0.1, hard=False, dim=-1)  # (b, s, n)
                    
                    mid_res = torch.einsum("bsd,nrd->bsnr", (dropout(x), lora_A_weight))
                    mid_res = torch.einsum("bsnr,ndr->bsnd", (mid_res, lora_B_weight))
                    lora_output = torch.einsum("bsnd,bsn->bsd", (mid_res, router_logits))

                else:
                    # 推理阶段，使用 Gumbel-Softmax，hard=True，使用 hard routing
                    router_probs = F.gumbel_softmax(router_logits, tau=0.1, hard=True, dim=-1)  # (b, s, n)
                    
                    # 由于 router_probs 是 one-hot 向量，我们可以直接获取索引
                    indices = torch.argmax(router_probs, dim=-1)  # (b, s)
                    
                    # 使用高级索引直接选择专家权重
                    selected_lora_A_weight = lora_A_weight[indices]  # (b, s, r, d)
                    selected_lora_B_weight = lora_B_weight[indices]  # (b, s, d, r)

                    selected_lora_A_output = torch.einsum("bsd,bsrd->bsr", (dropout(x), selected_lora_A_weight))  # (b, s, k)
                    lora_output = torch.einsum("bsr,bsdr->bsd", (selected_lora_A_output, selected_lora_B_weight))  # (b, s, d)
                result = result + lora_output * scaling
                
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moelora." + rep

