import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
from typing import Dict, Optional, Tuple


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        
        return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)


class GatingFusionMoEGate(nn.Module):
    def __init__(self, d_x: int, d_text: int, M: int = 4, K: int = 1, noisy_gating: bool = True):
        super().__init__()
        self.M = M
        self.k = K
        
        self.noisy_gating = noisy_gating

        self.gap = nn.AdaptiveAvgPool2d((1, 1))


        self.w_gate_x = nn.Parameter(torch.zeros(d_x, M), requires_grad=True)
        self.w_noise_x = nn.Parameter(torch.zeros(d_x, M), requires_grad=True)

        self.w_gate_text = nn.Parameter(torch.zeros(d_text, M), requires_grad=True)
        self.w_noise_text = nn.Parameter(torch.zeros(d_text, M), requires_grad=True)

        self.logit_weight_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        assert self.k <= self.M

    def forward(self, x: torch.Tensor, Degraded_feature: torch.Tensor, loss_coef=1e-2, noise_epsilon=1e-2):
        batch_size = x.shape[0]
        x_pooled = self.gap(x).view(batch_size, -1)
        
        clean_logits_x = x_pooled @ self.w_gate_x
        clean_logits_text = Degraded_feature @ self.w_gate_text
        
        if self.noisy_gating and self.training:
            raw_noise_stddev_x = x_pooled @ self.w_noise_x
            noise_stddev_x = self.softplus(raw_noise_stddev_x) + noise_epsilon
            noisy_logits_x = clean_logits_x + (torch.randn_like(clean_logits_x) * noise_stddev_x)
            
            raw_noise_stddev_text = Degraded_feature @ self.w_noise_text
            noise_stddev_text = self.softplus(raw_noise_stddev_text) + noise_epsilon
            noisy_logits_text = clean_logits_text + (torch.randn_like(clean_logits_text) * noise_stddev_text)
            
            alpha = torch.sigmoid(self.logit_weight_param)
            beta = 1.0 - alpha
            
            final_noise_stddev_squared = alpha.pow(2) * noise_stddev_x.pow(2) + beta.pow(2) * noise_stddev_text.pow(2)
            final_noise_stddev = torch.sqrt(final_noise_stddev_squared + 1e-8)
        else:
            noisy_logits_x = clean_logits_x
            noisy_logits_text = clean_logits_text
            
            final_noise_stddev = torch.full_like(clean_logits_x, noise_epsilon)

        if self.training:
             alpha = torch.sigmoid(self.logit_weight_param)
             beta = 1.0 - alpha
        else:
             with torch.no_grad():
                 alpha = torch.sigmoid(self.logit_weight_param)
                 beta = 1.0 - alpha
                    
        final_clean_logits = alpha * clean_logits_x + beta * clean_logits_text
        
        final_noisy_logits = alpha * noisy_logits_x + beta * noisy_logits_text
        
        logits = final_noisy_logits
        
        num_experts_to_consider = min(self.k + 1, self.M)
        top_logits, top_indices = logits.topk(num_experts_to_consider, dim=1)
        
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        
        zeros = torch.zeros_like(logits, requires_grad=False)
        
        gates = zeros.scatter(dim=1, index=top_k_indices, src=top_k_gates)
        
        importance = gates.sum(dim=0)
        
        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(final_clean_logits, final_noisy_logits, final_noise_stddev, top_logits)).sum(dim=0)
        else:
            load = self._gates_to_load(gates)
            
        loss = self.cv_squared(importance) + self.cv_squared(load)
        
        moe_loss = loss * loss_coef
        
        return gates, moe_loss

    def _gates_to_load(self, gates: torch.Tensor):
        return (gates > 0).sum(dim=0)

    def cv_squared(self, x: torch.Tensor):
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        
        variance = x.float().var()
        mean_squared = x.float().mean() ** 2
        
        return variance / (mean_squared + eps)

    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor):
        if noise_stddev is None or (noise_stddev == 0).all():
             logging.warning("`_prob_in_top_k` received zero or None noise_stddev. Load balancing might be inaccurate in eval.")
                
             return torch.zeros_like(clean_values)
        
        batch_size = clean_values.size(0)
        
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        
        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        
        is_in = torch.gt(noisy_values, threshold_if_in)
        
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        
        normal = Normal(self.mean, self.std)
        noise_stddev_safe = noise_stddev + 1e-8
        
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev_safe)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev_safe)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class FusedMoEGate(nn.Module):

    def __init__(self,
                 d_x: int,
                 d_text: int,
                 M: int = 4,
                 K: int = 1,
                 noisy_gating: bool = True,
                 fusion_dim: int = None,
                 calculate_standard_aux_loss: bool = False,
                 loss_coef: float = 1e-2,
                 noise_epsilon: float = 1e-2
                ):

        super().__init__()
        self.M = M
        self.k = K
        self.noisy_gating = noisy_gating
        self.calculate_standard_aux_loss = calculate_standard_aux_loss
        self.loss_coef = loss_coef
        self.noise_epsilon = noise_epsilon

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.input_fusion_dim = d_x + d_text
        if fusion_dim is None:
            fusion_dim = self.input_fusion_dim
        self.fusion_layer = nn.Linear(self.input_fusion_dim, fusion_dim)
        self.fusion_activation = nn.GELU()

        self.w_gate = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(fusion_dim, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M


    def forward(self, x: torch.Tensor, Degraded_feature: torch.Tensor):

        batch_size = x.shape[0]
        device = x.device
        x_pooled = self.gap(x).view(batch_size, -1)
        fused_feature = torch.cat([x_pooled, Degraded_feature], dim=1)
        if hasattr(self, 'fusion_layer'):
            fused_feature = self.fusion_layer(fused_feature)
            fused_feature = self.fusion_activation(fused_feature)
        clean_logits = fused_feature @ self.w_gate
        probs_for_mi = self.softmax(clean_logits)
        if self.noisy_gating and self.training:
            raw_noise_stddev = fused_feature @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
            noise_stddev = None
        num_experts_to_consider = min(self.k + 1, self.M)
        top_logits, top_indices = logits.topk(num_experts_to_consider, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=False)
        gates = zeros.scatter(dim=1, index=top_k_indices, src=top_k_gates)
        moe_loss = torch.tensor(0.0, device=device)
        if self.calculate_standard_aux_loss:
            importance = gates.sum(dim=0)
            if self.noisy_gating and self.k < self.M and self.training and noise_stddev is not None:
                load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(dim=0)
            else:
                load = self._gates_to_load(gates)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            moe_loss = loss * self.loss_coef
        return gates, moe_loss, probs_for_mi, top_k_indices

    def _gates_to_load(self, gates: torch.Tensor):
        return (gates > 0).sum(dim=0)

    def cv_squared(self, x: torch.Tensor):
        eps = 1e-10
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        variance = x.float().var()
        mean_squared = x.float().mean() ** 2
        return variance / (mean_squared + eps)

    def _prob_in_top_k(self, clean_values: torch.Tensor, noisy_values: torch.Tensor, noise_stddev: torch.Tensor, noisy_top_values: torch.Tensor):
        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        normal = Normal(self.mean, self.std)
        noise_stddev_safe = noise_stddev + 1e-6
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev_safe)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev_safe)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class SparseDispatcher(object):
    def __init__(self, num_experts: int, gates: torch.Tensor):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[:, 0][index_sorted_experts[:, 1]]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates) -> torch.Tensor:
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            if stitched.dim() == 4:
                 broadcast_gates = self._nonzero_gates.unsqueeze(-1).unsqueeze(-1)
            else:
                 broadcast_gates = self._nonzero_gates
                 for _ in range(stitched.dim() - self._nonzero_gates.dim()):
                     broadcast_gates = broadcast_gates.unsqueeze(-1)
            stitched = stitched.mul(broadcast_gates)
        output_dims = expert_out[-1].size()[1:]
        zeros = torch.zeros(
            self._gates.size(0),
            *output_dims,
            requires_grad=True,
            device=stitched.device,
            dtype=stitched.dtype
        )
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SimpleMoELayer(nn.Module):
    
    def __init__(self, input_channels: int, input_height: int, input_width: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.flattened_dim = input_channels * input_height * input_width
        self.gate = FusedMoEGate(d_x=input_channels, M=num_experts, K=top_k)
        self.experts = nn.ModuleList([
            nn.Linear(self.flattened_dim, self.flattened_dim) for i in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        
        assert x.dim() == 4 and x.shape[1] == self.input_channels and \
               x.shape[2] == self.input_height and x.shape[3] == self.input_width, \
               f"The input shape is incorrect~ Expected shape: [B={x.shape[0]}, C={self.input_channels}, H={self.input_height}, W={self.input_width}], but got {x.shape}"
        
        
        gates, moe_aux_loss = self.gate(x)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = []
        
        for i in range(self.num_experts):
            if expert_inputs[i].numel() > 0:
                current_input = expert_inputs[i]
                flattened_input = current_input.reshape(current_input.size(0), -1)
                processed_output_flat = self.experts[i](flattened_input)
                processed_output = processed_output_flat.reshape(
                    processed_output_flat.size(0),
                    self.input_channels,
                    self.input_height,
                    self.input_width
                )
                expert_outputs.append(processed_output)
            else:
                expert_outputs.append(None)
                
        valid_expert_outputs = []
        for i, output in enumerate(expert_outputs):
            if output is not None:
                valid_expert_outputs.append(output)

        if not valid_expert_outputs:
             print("Warning: No experts produced any output!")
             output = torch.zeros_like(x)
             return output, moe_aux_loss
            
        output = dispatcher.combine(valid_expert_outputs, multiply_by_gates=True)
        
        return output, moe_aux_loss
