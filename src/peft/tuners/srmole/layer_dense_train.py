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
from typing import Any, List, Optional, Union, Tuple
import torch.nn.functional as F
import math
import packaging
import torch
import transformers
from torch import nn
from torch.autograd import Variable  
from .idx_matmul_A import IndexedMatMul_A
from .idx_matmul_B import IndexedMatMul_B

from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
from .topk import TopK_custom
import time

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config



class SRMoLELayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "activate_r")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.activate_r = {}
        self.lora_router = nn.ParameterDict({})


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
        self.lora_router[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        # self.router[adapter_name] = TopkRouter(self.in_features, r, activate_r)
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
        self.expert_dropout_rate = 0.2

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
                
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_router = self.lora_router[active_adapter]
                activate_r = self.activate_r[active_adapter]
                r = self.r[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                # scaling = self.scaling[active_adapter]
                scaling = self.lora_alpha[active_adapter] / math.sqrt(r) if self.training \
                    else self.lora_alpha[active_adapter] / math.sqrt(activate_r)
                
                x_dropped = dropout(x)  # [batch_size, seq_len,in_features]
                logits = lora_router(x_dropped)

                if self.training:
                    gating_output =  F.softmax(logits, dim=-1) * r
                    mid = lora_A(dropout(x)) * gating_output
                    output = lora_B(mid)
                else: 

                    top_k_logits, indices = logits.topk(activate_r, dim=-1)
                    zeros = torch.full_like(logits, float('-inf'))
                    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
                    gating_output = F.softmax(sparse_logits, dim=-1)
                    gating_output = gating_output * activate_r

                    # flaten
                    x_dropped = x_dropped.view(-1, x_dropped.size(-1)) # [batch_size*seq_len, in_features]
                    gating_output = gating_output.view(-1, gating_output.size(-1)) # [batch_size*seq_len, r]
                    indices = indices.view(-1, indices.size(-1)).to(torch.int32) # [batch_size*seq_len, k]
                
                    # 使用 IndexedMatMul 进行计算
                    # 1. 准备权重
                    A = lora_A.weight  # [r, in_features]
                    B = lora_B.weight  # [out_features, r]

                    # 2. 第一步矩阵乘法：x @ A[indices]
                    intermediate = IndexedMatMul_A.apply(x_dropped.to(torch.float32), indices, A.to(torch.float32))  # [batch_size*seq_len, k]
                    
                    # 3. 应用 gating scores
                    selected_gates = torch.gather(gating_output, 1, indices.to(torch.int64))  # [batch_size*seq_len, k]
                    gated_intermediate = intermediate * selected_gates  # [batch_size*seq_len, k]
                    
                    # 4. 第二步矩阵乘法：(x @ A[indices]) @ B[:, indices]
                    output = IndexedMatMul_B.apply(gated_intermediate.to(torch.float32), indices.to(torch.int32), B.to(torch.float32))  # [batch_size*seq_len, out_features]

                    # unflatten
                    output = output.view(x.size(0), x.size(1), output.size(-1))
                    
                output = output.to(result.dtype)
                result = result + output * scaling
               
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "srmole." + rep