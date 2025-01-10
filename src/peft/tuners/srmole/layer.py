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
from pprint import pprint

if packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.33.0"):
    from transformers.integrations import deepspeed_config
else:
    from transformers.deepspeed import deepspeed_config

import math

def anneal_dropout_rate(initial_rate, final_rate, step, total_steps):
    """
    Dynamic dropout rate with cosine annealing.
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_steps))
    
    current_dropout_rate = final_rate + (initial_rate - final_rate) * cosine_decay

    return current_dropout_rate


class SRMoLELayer(LoraLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_router", "lora_biases")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout", "activate_r", "epsilon_greedy", "rank_partition")

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.activate_r = {}
        self.epsilon_greedy = {}
        self.rank_partition = {}
        self.lora_router = nn.ParameterDict({})
        self.lora_biases = nn.ParameterDict({})
        # self.lora_biases = {}


    def update_layer(self, adapter_name, r, activate_r, epsilon_greedy, rank_partition, lora_alpha, lora_dropout, init_lora_weights):
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
        self.epsilon_greedy[adapter_name] = epsilon_greedy
        self.rank_partition[adapter_name] = rank_partition
        self.lora_biases[adapter_name] = nn.Parameter(torch.zeros(r), requires_grad=False)

        # Actual trainable parameters
        # Right singular vectors
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

        # The current rank
        self.lora_router[adapter_name] = nn.Linear(self.in_features, r // rank_partition, bias=False)
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
        epsilon_greedy: bool = False,
        rank_partition: int = 1, 
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool = True,
        initial_expert_dropout_rate: float = 0.6,
        dropout_anneal_steps: int = 1594,
        layer_idx: str = None,
        bias_update_rate: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__()
        SRMoLELayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, activate_r, epsilon_greedy, rank_partition, lora_alpha, lora_dropout, init_lora_weights)
        # self.soft_topk = TopK_custom(activate_r)
    
        ## Load balancing parameters
        self.bias_update_rate = bias_update_rate  # Update rate 'u'
        self.num_experts = r  # Number of experts       
        
        # dynamic_expert_dropout
        self.dropout_anneal_steps = dropout_anneal_steps 
        self.current_step = 0
        self.initial_expert_dropout_rate = initial_expert_dropout_rate
        self.final_expert_dropout_rate = 0.0
        
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # update expert dropout rate
        # if self.training:
        #     # update_expert_drop_rate
        #     if self.current_step < self.dropout_anneal_steps:
        #         self.expert_dropout_rate = anneal_dropout_rate(
        #             self.initial_expert_dropout_rate, self.final_expert_dropout_rate,
        #             self.current_step, self.dropout_anneal_steps
        #         )
        #     else:
        #         self.expert_dropout_rate = self.final_expert_dropout_rate
        #     # print(f"current_step: {self.current_step}")
        #     # print(f"expert_dropout_rate: {self.expert_dropout_rate}")
        #     self.current_step += 1
        
        
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
                lora_biases = self.lora_biases[active_adapter]
                activate_r = self.activate_r[active_adapter]
                epsilon_greedy = self.epsilon_greedy[active_adapter]
                rank_partition = self.rank_partition[active_adapter]
                r = self.r[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]


                x_dropped = dropout(x)  # [batch_size, seq_len,in_features]
                logits = lora_router(x_dropped)
                logits = logits.repeat_interleave(rank_partition, dim=-1)
        
                # Expand lora_biases to match logits shape [batch_size, seq_len, num_experts]
                lora_biases_expanded = lora_biases.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_experts]
                logits = logits + lora_biases_expanded.to(logits.device)
                
                # # drop
                # if self.training and self.current_step < self.dropout_anneal_steps and self.expert_dropout_rate > 0.0:
                #     # drop expert
                #     dropout_mask = torch.bernoulli(torch.ones_like(logits) * (1 - self.expert_dropout_rate)).to(logits.device)
                #     logits = logits.masked_fill(dropout_mask == 0, float('-inf'))

                if not epsilon_greedy or not self.training:
                    # 直接使用 topk
                    # print("topk")
                    top_k_logits, indices = logits.topk(activate_r, dim=-1)
                else:
                    # epsilon-greedy 策略
                    if torch.rand(1).item() < 0.1:  # 0.01 的概率随机选择
                        # print("epsilon-greedy")
                        batch_size, seq_len, r = logits.shape
                        indices = torch.randint(0, r, (batch_size, seq_len, activate_r), device=logits.device)
                        top_k_logits = torch.gather(logits, dim=-1, index=indices)
                    else:
                        # print("topk")
                        top_k_logits, indices = logits.topk(activate_r, dim=-1)
                
                # if not self.training and hasattr(self, 'layer_idx'):
                #     with open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/expert_routing.txt", "a") as f:
                #         f.write(f"Layer: {self.layer_idx}\n")
                
                #     with open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouyixiao-240108120127/work/acl-2025-moe-lora/LLaMA-Factory/results/expert_routing.txt", "a") as f:
                #         for i in range(indices.size(0)):  # batch_size
                #             for j in range(indices.size(1)):  # seq_len
                #                 for k in range(indices.size(2)):  # top_k
                #                     f.write(str(indices[i][j][k].item()) + " ")
                #                 f.write("\n")
                #             f.write("\n")
                    
                    # # 将indices展平成一维张量
                    # flat_indices = indices.view(-1)

                    # # 使用bincount统计每个专家的使用次数，确保长度至少为64
                    # counts = torch.bincount(flat_indices, minlength=64).tolist()

                    # # 将计数结果写入文件
                    # with open("expert_routing.txt", "a") as f:
                    #     f.write(str(counts) + "\n")
                
                
                # Update biases based on expert assignments
                with torch.no_grad():
                    expert_counts = torch.bincount(
                        indices.view(-1),
                        minlength=self.num_experts
                    ).float()  # [num_experts]
                    avg_count = expert_counts.mean()  # 标量
                    e_i = avg_count - expert_counts  # [num_experts]
                    lora_biases += self.bias_update_rate * e_i.sign()
               
                    self.lora_biases[active_adapter] = lora_biases
                    
                        
                        
                zeros = torch.full_like(logits, float('-inf'))
                sparse_logits = zeros.scatter(-1, indices, top_k_logits)
                gating_output = F.softmax(sparse_logits, dim=-1)
                gating_output = gating_output * activate_r
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