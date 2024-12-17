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
import time

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear =nn.Linear(n_embed, num_experts)
       
    def forward(self, x):
        logits = self.linear(x)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


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
        self.router[adapter_name] = TopkRouter(self.in_features, r, activate_r)
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
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]

                                # 直接计算
                start_time = time.time()
                direct_result = result + lora_B(lora_A(dropout(x))) * scaling
                direct_time = time.time() - start_time


                start_time = time.time()

                # Reshape inputs for batch processing
                flat_x = dropout(x).view(-1, x.size(-1))
                gating_output, indices = router(x)  # 形状为 (b, s, r)

                final_output = torch.zeros_like(x)

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
                        final_output[expert_mask] += weighted_output.squeeze(1)
                
                result = result + final_output * scaling
                srmole_time = time.time() - start_time
                print("start_time: {}; direct_time: {}; srmole_time: {}".format(start_time, direct_time, srmole_time))
        
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep