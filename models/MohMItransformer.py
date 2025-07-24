from typing import Tuple, Dict

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

        
        gate_logits = self.gating_network(text_feature)  

        if training:
            
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise  

        gate_weights = F.softmax(gate_logits, dim=-1) 

        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  

        moe_output = torch.zeros_like(x) 

        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  
            weight = topk_values[:, i].view(B, 1, 1, 1)  

            for j in range(self.num_experts):
                mask = (expert_idx == j).view(B, 1, 1, 1)  
                if mask.any():
                    moe_output += weight * mask * self.experts[j](x)  

        return moe_output

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

class MoH_MDTA_Attention_TaskMI(nn.Module):
   
    def __init__(self, dim: int, num_heads: int = 8, top_k_attn: int = 2,
                 bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.,
                 w_MI_attn: float = 0.01, epsilon: float = 1e-7,
                 calculate_standard_aux_loss: bool = False, # <-- Add flag
                 loss_coef_attn: float = 1e-2):           # <-- Add coef for attn LB loss
        super().__init__()
        assert dim % num_heads == 0; assert top_k_attn <= num_heads
        self.num_heads = num_heads; self.head_dim = dim // num_heads
        self.top_k_attn = top_k_attn; self.scale = self.head_dim ** -0.5
        self.w_MI_attn = w_MI_attn; self.epsilon = epsilon
        self.calculate_standard_aux_loss = calculate_standard_aux_loss # Store flag
        self.loss_coef_attn = loss_coef_attn                 # Store coef

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop); self.proj_drop = nn.Dropout(proj_drop)
        self.router_task_main = nn.Conv2d(dim, num_heads, 1, bias=False)
        self.router_task_aux = nn.Conv2d(dim, num_heads, 1, bias=False)

    # --- CV^2 Helper ---
    def cv_squared(self, x: torch.Tensor):
        eps = 1e-10
        if x.shape[0] <= 1: return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x.float().var() / (x.float().mean()**2 + eps)

    # --- MI Loss Calculation (Unchanged) ---
    def _calculate_batch_mi_loss_attn(self, probs_attn_flat: torch.Tensor, task_selector_flat: torch.Tensor) -> torch.Tensor:
        if self.w_MI_attn <= 0 or not self.training:
            return torch.tensor(0.0, device=probs_attn_flat.device)
        batch_size_flat, num_heads = probs_attn_flat.shape
        p_head_batch_flat = probs_attn_flat.mean(dim=0)
        total_mi = 0.0
        tasks_present = torch.unique(task_selector_flat)
        for task_id_val in tasks_present:
            task_mask_flat = (task_selector_flat == task_id_val)
            num_task_samples_flat = task_mask_flat.sum()
            if num_task_samples_flat == 0: continue
            p_t_batch_flat = num_task_samples_flat / batch_size_flat
            p_head_given_t_batch_flat = probs_attn_flat[task_mask_flat].mean(dim=0)
            kl_div_term = p_head_given_t_batch_flat * torch.log(
                p_head_given_t_batch_flat / (p_head_batch_flat + self.epsilon) + self.epsilon
            )
            kl_div = kl_div_term.sum()
            total_mi += p_t_batch_flat * kl_div
        mi_loss_attn = -self.w_MI_attn * total_mi
        return mi_loss_attn

    def forward(self, x: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: # Return Dict
        B, C, H, W = x.shape; N = H * W

        # 1. Routing
        router_module = self.router_task_main if task_id == 0 else self.router_task_aux
        router_logits = router_module(x) # [B, num_heads, H, W]
        router_logits_flat = rearrange(router_logits, 'b head h w -> (b h w) head') # [B*N, num_heads]
        probs_attn_flat = F.softmax(router_logits_flat, dim=1) # [B*N, num_heads]

        # 2. MI Loss (Unconditional during training)
        task_selector_single = torch.full((B,), task_id, dtype=torch.long, device=x.device)
        task_selector_flat = task_selector_single.repeat_interleave(N)
        mi_loss_attn = self._calculate_batch_mi_loss_attn(probs_attn_flat, task_selector_flat)

        # 3. Top-K Selection & Gate Calculation
        with torch.no_grad():
             _, indices_attn = torch.topk(probs_attn_flat, k=self.top_k_attn, dim=1) # [B*N, k_attn]
             mask_attn_flat = F.one_hot(indices_attn, num_classes=self.num_heads).sum(dim=1).float() # [B*N, num_heads] binary

        masked_gates_flat = probs_attn_flat * mask_attn_flat # Apply mask
        denom_s = torch.sum(masked_gates_flat, dim=1, keepdim=True)
        denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
        normalized_masked_gates_flat = masked_gates_flat / denom_s # [B*N, num_heads], normalized over K selected heads

        # 4. Conditional Load Balancing Loss Calculation
        attn_load_balancing_loss = torch.tensor(0.0, device=x.device)
        if self.calculate_standard_aux_loss and self.training:
            # Importance: Sum of normalized gate values per head over all tokens
            importance_attn = normalized_masked_gates_flat.sum(dim=0) # Shape [num_heads]
            # Load: Sum of probabilities per head over all tokens
            load_attn = probs_attn_flat.sum(dim=0) # Shape [num_heads]

            loss = self.cv_squared(importance_attn) + self.cv_squared(load_attn)
            attn_load_balancing_loss = loss * self.loss_coef_attn

        # 5. Apply Gating to Attention Output
        masked_gates_attn = rearrange(normalized_masked_gates_flat, '(b h w) head -> b head 1 (h w)', b=B, h=H, w=W)

        # --- MDTA ---
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (h c) H W -> b h c (H W)', h=self.num_heads) # Use H,W from outer scope
        k = rearrange(k, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        v = rearrange(v, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v) # [B, head, c, N]

        # Apply gating
        gated_out = out * masked_gates_attn * self.top_k_attn # Scale by K

        # Reshape back
        out = rearrange(gated_out, 'b head c (h w) -> b (head c) h w', h=H, w=W)
        out = self.project_out(out)
        out = self.proj_drop(out)

        # 6. Prepare Loss Dictionary
        aux_losses_attn = {
            'attn_load_balancing_loss': attn_load_balancing_loss,
            'attn_mi_loss': mi_loss_attn
        }

        return out, aux_losses_attn # Return dict


class MMoEFeedForward(nn.Module):
    # ... (keep __init__ and _calculate_batch_mi_loss as corrected before) ...
    def __init__(self, dim, d_text=512, num_experts=16, top_k=2,
                 w_MI=0.01, epsilon=1e-7,
                 calculate_standard_aux_loss: bool = False,
                 loss_coef: float = 1e-2,
                 noisy_gating: bool = True,
                 **expert_kwargs):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        self.w_MI = w_MI
        self.epsilon = epsilon

        gate_init_args = {
            'd_x': dim, 'd_text': d_text, 'M': num_experts, 'K': top_k,
            'calculate_standard_aux_loss': calculate_standard_aux_loss,
            'loss_coef': loss_coef, 'noisy_gating': noisy_gating
        }
        self.gate_task_main = FusedMoEGate(**gate_init_args)
        self.gate_task_aux = FusedMoEGate(**gate_init_args)

        # Experts (using kwargs)
        self.experts = nn.ModuleList()
        downsample_mode = expert_kwargs.get('downsample_mode', 'max')
        for i in range(num_experts):
            if i % 3 == 0: expert = NoRescaleExpert(dim=dim)
            elif i % 3 == 1: expert = UpThenDownExpert(dim=dim, downsample_mode=downsample_mode)
            else: expert = DownThenUpExpert(dim=dim, downsample_mode=downsample_mode)
            self.experts.append(expert)

    def _calculate_batch_mi_loss(self, probs: torch.Tensor, task_selector_flat: torch.Tensor):
        # --- (Keep this method as is) ---
        if self.w_MI <= 0 or not self.training:
            return torch.tensor(0.0, device=probs.device)
        batch_size, num_experts = probs.shape
        p_e_batch = probs.mean(dim=0)
        total_mi = 0.0
        tasks_present = torch.unique(task_selector_flat)
        for task_id_val in tasks_present:
            task_mask = (task_selector_flat == task_id_val)
            num_task_samples = task_mask.sum()
            if num_task_samples == 0: continue
            p_t_batch = num_task_samples / batch_size
            p_e_given_t_batch = probs[task_mask].mean(dim=0)
            kl_div_term = p_e_given_t_batch * torch.log(
                p_e_given_t_batch / (p_e_batch + self.epsilon) + self.epsilon
            )
            kl_div = kl_div_term.sum()
            total_mi += p_t_batch * kl_div
        mi_loss = -self.w_MI * total_mi
        return mi_loss

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor, task_id: int):
        batch_size, channels, height, width = x.shape # Get dimensions
        assert channels == self.dim
        task_selector_flat = torch.full((batch_size,), task_id, dtype=torch.long, device=x.device)

        gate_module = self.gate_task_main if task_id == 0 else self.gate_task_aux
        gate_weights, standard_moe_loss, probs, indices = gate_module(x, text_feature)
        batch_mi_loss = self._calculate_batch_mi_loss(probs, task_selector_flat)

        dispatcher = SparseDispatcher(self.num_experts, gate_weights)
        expert_inputs = dispatcher.dispatch(x)

        expert_outputs = [] # Initialize empty list to append to

        # Define expected output shape (without batch dim)
        output_dims = (self.dim, height, width)

        for i in range(self.num_experts):
            expert_input_i = expert_inputs[i]
            if expert_input_i is not None and expert_input_i.shape[0] > 0:
                # Compute output if input exists
                expert_output_i = self.experts[i](expert_input_i)
                expert_outputs.append(expert_output_i)
            else:
                # Create an empty tensor with correct shape, device, dtype
                empty_output = torch.zeros(0, *output_dims, device=x.device, dtype=x.dtype)
                expert_outputs.append(empty_output) # Append empty tensor instead of None

        # Combine should now work as expert_outputs contains only Tensors
        output = dispatcher.combine(expert_outputs, multiply_by_gates=True)

        aux_losses = {
            'moe_load_balancing_loss': standard_moe_loss,
            'moe_mi_loss': batch_mi_loss
        }
        return output, aux_losses

class HeterogeneousMoH_TaskMI_TransformerBlock(nn.Module):
    # ... (init signature remains the same) ...
    def __init__(self, dim: int, num_heads: int = 8, top_k_attn: int = 2, # MoH params
                 d_text: int = 512, num_experts: int = 8, top_k_ffn: int = 2,   # MoE params
                 bias: bool = False, LayerNorm_type='WithBias', # General params
                 eps: float = 1e-6, attn_drop: float = 0., proj_drop: float = 0., # Dropout/Norm
                 w_MI_attn: float = 1, w_MI_ffn: float = 1,              # MI weights
                 calculate_standard_aux_loss: bool = False,                      # Flag for BOTH layers
                 loss_coef_attn: float = 1e-2, loss_coef_ffn: float = 1e-2,    # LB coefficients
                 noisy_gating_ffn: bool = True,                                # FFN noisy gating
                 **expert_kwargs):                                            # FFN Expert args
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim, eps=eps)
        self.norm2 = nn.GroupNorm(1, dim, eps=eps)

        self.attn = MoH_MDTA_Attention_TaskMI( # Or _ImgOnly if that's the one
            dim=dim, num_heads=num_heads, top_k_attn=top_k_attn,
            bias=bias, attn_drop=attn_drop, proj_drop=proj_drop,
            w_MI_attn=w_MI_attn, epsilon=eps,
            calculate_standard_aux_loss=calculate_standard_aux_loss,
            loss_coef_attn=loss_coef_attn
        )

        # This call should now work correctly
        self.moe_ffn = MMoEFeedForward(
            dim=dim, d_text=d_text, num_experts=num_experts, top_k=top_k_ffn,
            w_MI=w_MI_ffn, epsilon=eps,
            # Pass the arguments explicitly:
            calculate_standard_aux_loss=calculate_standard_aux_loss,
            loss_coef=loss_coef_ffn,
            noisy_gating=noisy_gating_ffn,
            **expert_kwargs # Pass down expert arguments
        )

    def forward(self, x: torch.Tensor, text_feature: torch.Tensor, task_id: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual1 = x
        x_norm1 = self.norm1(x)
        attn_output, attn_losses_dict = self.attn(x_norm1, task_id=task_id)
        x = residual1 + attn_output

        residual2 = x
        x_norm2 = self.norm2(x)
        # Ensure the key 'moe_mi_loss' is returned here
        moe_output, moe_losses_dict = self.moe_ffn(x_norm2, text_feature, task_id=task_id)
        x = residual2 + moe_output

        # Combine losses (check keys)
        total_mi_loss = attn_losses_dict['attn_mi_loss'] + moe_losses_dict['moe_mi_loss'] # Check this key name
        total_load_balancing_loss = attn_losses_dict['attn_load_balancing_loss'] + moe_losses_dict['moe_load_balancing_loss']

        combined_aux_losses = {
            'total_mi_loss': total_mi_loss,
            'total_load_balancing_loss': total_load_balancing_loss
        }
        return x, combined_aux_losses
