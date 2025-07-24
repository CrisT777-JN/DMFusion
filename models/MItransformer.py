from typing import Tuple

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers
from models.FMoEGate import FusedMoEGate,SparseDispatcher,GatingFusionMoEGate

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False): 
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Mlp(nn.Module):
    

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.0, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=1.0, bias=True, LayerNorm_type='BiasFree'):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FFTAttentionLite(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 1)  

    def forward(self, x):
        
        x = x.float()

        
        x_fft = torch.fft.rfft2(x, norm='ortho')

        
        x_mag = torch.abs(x_fft)
        gate = torch.tanh(self.proj(x_mag))  

        x_fft = x_fft * gate 

        x_out = torch.fft.irfft2(x_fft, s=x.shape[-2:], norm='ortho')
        return x_out.to(x.dtype)  


class ExpertDilatedConv(nn.Module):
    
    def __init__(self, dim, hidden_dim_factor=4, dilation=1):
        super().__init__()
        hidden_dim = int(dim * hidden_dim_factor)
        # Best practice: Expand channels, process, then contract back
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1), # Pointwise expansion
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden_dim if hidden_dim % dim == 0 and hidden_dim >= dim else 1), # Depthwise or Grouped Conv with dilation
            # Use standard conv if grouping is not straightforward
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1)  # Pointwise contraction
        )

    def forward(self, x):
        return self.net(x) + x # Residual connection

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=1, noise_std=0.1):
        super(MoEFeedForward, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std  

        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])

        
        self.gating_network = nn.Linear(512, num_experts, bias=False)

    def forward(self, x, text_feature, training=True):
        B, C, H, W = x.shape  

        
        gate_logits = self.gating_network(text_feature)  # (B, num_experts)

        if training:
            
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise  

        gate_weights = F.softmax(gate_logits, dim=-1)  

        
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B, top_k)

        
        moe_output = torch.zeros_like(x)  # (B, C, H, W)

        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  
            weight = topk_values[:, i].view(B, 1, 1, 1)  

            
            for j in range(self.num_experts):
                mask = (expert_idx == j).view(B, 1, 1, 1)  
                if mask.any():
                    moe_output += weight * mask * self.experts[j](x) 

        return moe_output

class MTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(MTransformerBlock, self).__init__()

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads)

        
        self.moe_ffn = MoEFeedForward(dim)

    def forward(self, x, text_feature):
        
        x = x + self.attn(x)
        x = x + self.moe_ffn(x, text_feature)

        return x

class SparseMoEFeedForward(nn.Module): 
    
    def __init__(self, dim, d_text=512, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        
        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])

    
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        
        gates, moe_aux_loss = self.gate(x, text_feature) 
         
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [None] * self.num_experts
        for i in range(self.num_experts):
            if expert_inputs[i] is not None and expert_inputs[i].shape[0] > 0:
                
                expert_outputs[i] = self.experts[i](expert_inputs[i])

        valid_expert_outputs = [output for output in expert_outputs if output is not None]
        if not valid_expert_outputs:
            print("Warning: No experts produced any output! Returning a zero tensor!")
            output = torch.zeros_like(x)
        else:
            output = dispatcher.combine(valid_expert_outputs, multiply_by_gates=True)

        
        return output, moe_aux_loss

class HomogeneousSparseMoEFeedForward(nn.Module): 
    
    def __init__(self, dim, d_text=512, num_experts=16, top_k=2,expert_use_fft = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim

        
        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        gates, moe_aux_loss = self.gate(x, text_feature)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x) 
        
        expert_outputs = [] 

        
        output_dims = (self.dim, height, width)

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output)

        
        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)
        

        return output, moe_aux_loss

class SparseTransformerBlock(nn.Module):
    
    
    def __init__(self, dim, d_text=512, num_heads=4, num_experts=16, top_k=2, bias=False, LayerNorm_type='WithBias', eps=1e-6): 
        super().__init__()
        
        self.norm1 = nn.GroupNorm(1, dim, eps=eps) 
        self.norm2 = nn.GroupNorm(1, dim, eps=eps) 

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads) 
        self.moe_ffn = HomogeneousSparseMoEFeedForward(dim, d_text, num_experts=num_experts, top_k=top_k)

    
    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        
        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1)
        x = residual1 + attn_output 

        
        residual2 = x
        x_norm2 = self.norm2(x)
        moe_output, moe_loss = self.moe_ffn(x_norm2, text_feature) 
        x = residual2 + moe_output 

        return x, moe_loss 


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out

class SimpleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out

def create_learnable_block(dim):
    return SimpleBlock(dim)
# --- Modified Heterogeneous Experts with Learnable Block ---

class NoRescaleExpert(nn.Module):
    
    def __init__(self, dim): # Requires dim for Conv2d
        super().__init__()
        # Non-parameterized operation
        # Learnable convolutional block
        self.learnable_block = create_learnable_block(dim)

    def forward(self, x):
        # Apply non-parameterized op first, then the learnable block
        x = self.learnable_block(x)
        return x


class UpThenDownExpert(nn.Module):
    
    def __init__(self, dim, downsample_mode='max'): # Requires dim
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Learnable block applied at the upsampled resolution
        self.learnable_block = create_learnable_block(dim)
        if downsample_mode == 'max':
            self.downsample_net = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'avg':
            self.downsample_net = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("downsample_mode must be 'max' or 'avg'")

    def forward(self, x):
        x = self.upsample(x)         # 1. Up-sample
        x = self.learnable_block(x)  # 2. Apply learnable block
        x = self.downsample_net(x)   # 3. Down-sample
        return x


class DownThenUpExpert(nn.Module):
    
    def __init__(self, dim, downsample_mode='max'): # Requires dim
        super().__init__()
        if downsample_mode == 'max':
            self.downsample_net = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample_mode == 'avg':
            self.downsample_net = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("downsample_mode must be 'max' or 'avg'")
        # Learnable block applied at the downsampled resolution
        self.learnable_block = create_learnable_block(dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.downsample_net(x)   # 1. Down-sample
        x = self.learnable_block(x)  # 2. Apply learnable block
        x = self.upsample(x)         # 3. Up-sample
        return x


class MultiScaleHeterogeneousMoEFeedForward(nn.Module):
    

    def __init__(self, dim, d_text=512, num_experts=16, top_k=2, expert_use_fft=True, downsample_mode='max'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim  

        self.gate = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k)

        
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 3 == 0:
                
                expert = NoRescaleExpert(dim=dim)
            elif i % 3 == 1:
                
                expert = UpThenDownExpert(dim=dim, downsample_mode=downsample_mode)
            else:
                
                expert = DownThenUpExpert(dim=dim, downsample_mode=downsample_mode)
            self.experts.append(expert)

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor):
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        gates, moe_aux_loss = self.gate(x, text_feature)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)  

        expert_outputs = []
        output_dims = (self.dim, height, width)  

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output)

        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        return output, moe_aux_loss

class MMoEFeedForward(nn.Module):
    
    def __init__(self, dim, d_text=512, num_experts=16, top_k=2, expert_use_fft=True, downsample_mode='max',
                 w_MI=0.01, # Keep MI weight, will be used directly in batch loss
                 num_tasks=2,
                 epsilon=1e-7): # Small constant for numerical stability
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        self.num_tasks = num_tasks
        self.w_MI = w_MI
        self.epsilon = epsilon

        # Use FusedMoEGate: Assumes it returns (gates, moe_loss, probs, indices)
        # where probs are the clean probabilities before top-k
        self.gate_task_main = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k,calculate_standard_aux_loss=False)
        self.gate_task_aux = FusedMoEGate(d_x=dim, d_text=d_text, M=num_experts, K=top_k,calculate_standard_aux_loss=False)

        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i % 3 == 0:
                expert = NoRescaleExpert(dim=dim)
            elif i % 3 == 1:
                expert = UpThenDownExpert(dim=dim, downsample_mode=downsample_mode)
            else:
                expert = DownThenUpExpert(dim=dim, downsample_mode=downsample_mode)
            self.experts.append(expert)
       

    def _calculate_batch_mi_loss(self, probs: torch.Tensor, task_selector_flat: torch.Tensor):
        
        if self.w_MI <= 0 or not self.training:
            return torch.tensor(0.0, device=probs.device)

        batch_size, num_experts = probs.shape

        # Estimate P(E) batch = Average probability for each expert over the whole batch
        p_e_batch = probs.mean(dim=0) # Shape: [M]

        # Estimate P(T) batch and P(E|T) batch
        total_mi = 0.0
        tasks_present = torch.unique(task_selector_flat)

        for task_id_val in tasks_present:
            task_mask = (task_selector_flat == task_id_val)
            num_task_samples = task_mask.sum()

            if num_task_samples == 0:
                continue

            # Estimate P(T=i)_batch = fraction of samples in batch with task_id i
            p_t_batch = num_task_samples / batch_size

            # Estimate P(E|T=i)_batch = Average probability for each expert over samples of task_id i
            p_e_given_t_batch = probs[task_mask].mean(dim=0) # Shape: [M]

            # Calculate KL divergence KL( P(E|T=i)_batch || P(E)_batch )
            # KL = sum P(E|T=i) * log( P(E|T=i) / P(E) )
            kl_div_term = p_e_given_t_batch * torch.log(
                p_e_given_t_batch / (p_e_batch + self.epsilon) + self.epsilon
            )
            kl_div = kl_div_term.sum()

            # Add weighted KL to total MI approximation
            total_mi += p_t_batch * kl_div

        # Final loss is negative weighted MI approximation
        mi_loss = -self.w_MI * total_mi
        return mi_loss


    def forward(self, x: torch.Tensor, text_feature: torch.Tensor, task_id=0):
        
        batch_size, channels, height, width = x.shape
        assert channels == self.dim

        task_selector_flat = torch.full((batch_size,), task_id, dtype=torch.long, device=x.device)

        # --- Gate Selection ---
        if task_id == 0:
            gate_module = self.gate_task_main
        elif task_id == 1:
            gate_module = self.gate_task_aux
        else:
            raise ValueError("Invalid task_id")

        # --- Get Gate Outputs ---
        # Assumes FusedMoEGate returns: sparse_gates, standard_aux_loss, clean_probs, top_k_indices
        gate_weights, standard_moe_loss, probs, indices = gate_module(x, text_feature)

        # --- Calculate Batch MI Loss ---
        # Uses the CLEAN probabilities returned by the gate
        batch_mi_loss = self._calculate_batch_mi_loss(probs, task_selector_flat)

        # --- Sparse Dispatch and Expert Computation ---
        dispatcher = SparseDispatcher(self.num_experts, gate_weights)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = []
        output_dims = (self.dim, height, width)

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output)

        # --- Combine Outputs ---
        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)


        return output, batch_mi_loss # Return aux_losses dict
class HeterogeneousTransformerBlock(nn.Module):

    def __init__(self, dim, d_text=512, num_heads=4, num_experts=16, top_k=2, bias=False, LayerNorm_type='WithBias',
                 eps=1e-6):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(1, dim, eps=eps)
        self.norm2 = nn.GroupNorm(1, dim, eps=eps)

        self.attn = Attention(dim=dim, bias=bias, num_heads=num_heads)  
        self.moe_ffn = MMoEFeedForward(dim, d_text, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor, task_id: int):
        
        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1)
        x = residual1 + attn_output  

        residual2 = x
        x_norm2 = self.norm2(x)
        moe_output, moe_loss = self.moe_ffn(x_norm2, text_feature, task_id=task_id)  
        x = residual2 + moe_output  

        return x, moe_loss  
