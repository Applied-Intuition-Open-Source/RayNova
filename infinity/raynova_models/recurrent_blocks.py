# Copyright (c) 2026 Applied Intuition, Inc.
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

"""
Definitions of blocks of VAR transformer model.
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union

from cycler import K

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, get_dropout_layer, FFN, FFNSwiGLU, CrossAttention, apply_rotary_emb
from infinity.raynova_models.rope import apply_rotary_emb_with_camera, apply_rope_condition, apply_rotary_emb_with_camera_ray6D
from infinity.raynova_models.stream_blocks import RoPECrossAttention, MVSelfAttention

# Import flash_attn's fused ops
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight

class RecurrentSelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, use_flex_attn=False,
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
        input_dim=None, time_chunk=None,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.time_chunk = time_chunk
        if input_dim is None:
            input_dim = embed_dim
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, self.num_heads, 1, 1)
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(input_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, input_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cache = None

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def recurrent_caching(self, enable: bool): # cache past frame features during training
        self.caching = enable
        self.cache = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=-1, view_meta_data=None, timesteps=None, num_views=None, past_cache=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector: block-wise, lower-triangle matrix where 0 means visible and -inf means invisible.
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, q_L, _ = x.shape
        C = self.embed_dim
        
        if past_cache is None:
            past_cache = x.detach().to(torch.bfloat16)
        else:
            assert past_cache.shape[1] % q_L == 0, 'past_cache must be divisible by q_L'
            x = torch.cat((past_cache, x), dim=1)
            past_cache = x.detach().to(torch.bfloat16)
            if past_cache.shape[1] > self.time_chunk*q_L:
                past_cache = past_cache[:, -self.time_chunk*q_L:]
            
        L = x.shape[1]
        
        assert L % q_L == 0, 'L must be divisible by q_L'
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        if view_meta_data is not None:
            if '6D' in view_meta_data['view_embed_type']:
                q, k = apply_rotary_emb_with_camera_ray6D(q, k, view_meta_data=view_meta_data) #, freqs_cis=freqs_cis)
            else:
                q, k = apply_rotary_emb_with_camera(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views) #, freqs_cis=freqs_cis)
        elif rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind, num_views=num_views) #, freqs_cis=freqs_cis)

        if q_L != L:
            if attn_bias_or_two_vector is not None:
                attn_bias_or_two_vector = torch.cat([attn_bias_or_two_vector]*(L//q_L), dim=-1)
                q = q[:, :, -q_L:]

        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            q = q.type_as(v)
            k = k.type_as(v)
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, q_L, C)

        return self.proj_drop(self.proj(oup)), past_cache

    def extra_repr(self) -> str:
        return f'tau={self.tau}, cos_attn={self.cos_attn}'


class RecurrentAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        use_temporal_attn=True, use_condition_rope=False,
        time_chunk=None, use_local_attn=False,
    ):
        super(RecurrentAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_temporal_attn = use_temporal_attn
        self.use_local_attn = use_local_attn


        if use_local_attn:
            self.sa = MVSelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            )
        if use_temporal_attn:
            self.sa_temporal = RecurrentSelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                time_chunk=time_chunk
            )
        self.ca = RoPECrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn, use_condition_rope=use_condition_rope)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
            if use_temporal_attn:
                self.ada_gss_temporal = nn.Parameter(torch.randn(1, 1, 3, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 3*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
            if use_temporal_attn:
                lin_temporal = nn.Linear(cond_dim, 3*embed_dim)
                self.ada_lin_temporal = nn.Sequential(nn.SiLU(inplace=False), lin_temporal) if act else nn.Sequential(lin_temporal)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only

    def _sequence_to_scale_reshape(self, x_sa, num_views, scale_schedule):
        len_mv_sf = np.sum([np.prod(item) for item in scale_schedule]) * num_views
        bs = x_sa.shape[0]
        seq_len = x_sa.shape[1]
        num_channels = x_sa.shape[-1]
        full_frames = seq_len // len_mv_sf

        if not self.training:
            x_sa = x_sa.reshape(bs*num_views, -1, num_channels)
            return x_sa

        x_sa_full = x_sa.reshape(bs, full_frames, len_mv_sf, num_channels)

        x_sa_full_list = []
        start_id = 0
        for scale in scale_schedule:
            end_id = start_id + np.prod(scale) * num_views
            x_sa_full_scale = x_sa_full[:, :, start_id:end_id].reshape(bs, full_frames, num_views, np.prod(scale), num_channels)
            x_sa_full_scale = x_sa_full_scale.flatten(1, 2)
            x_sa_full_list.append(x_sa_full_scale)
            
            start_id = end_id

        x_sa_full = torch.cat(x_sa_full_list, dim=2)
        x_sa_full = x_sa_full.flatten(0, 1)
                
        return x_sa_full

    def _scale_to_sequence_reshape(self, x_sa, num_views, bs, scale_schedule):
        len_sv_sf = np.sum([np.prod(item) for item in scale_schedule])
        full_frames = x_sa.shape[0] // bs // num_views

        num_channels = x_sa.shape[-1]

        if not self.training:
            x_sa = x_sa.reshape(bs, -1, num_channels)
            return x_sa

        x_sa_full = x_sa.reshape(bs, full_frames, num_views, len_sv_sf, num_channels)
        
        x_sa_full_list = []
        start_id = 0
        for scale in scale_schedule:
            end_id = start_id + np.prod(scale)

            x_sa_full_scale = x_sa_full[:, :, :, start_id:end_id].flatten(2, 3)
            x_sa_full_list.append(x_sa_full_scale)

            start_id = end_id

        x_sa_full = torch.cat(x_sa_full_list, dim=2).flatten(1, 2)

        return x_sa_full

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=-1, num_views=None, timesteps=None, view_meta_data=None, cond_BD_temporal=None,past_cache=None):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        assert self.training, 'RecurrentAttnBlock is only used during training'
        with torch.cuda.amp.autocast(enabled=False):    # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
                if self.use_temporal_attn:
                    gamma_temporal, scale_temporal, shift_temporal = (self.ada_gss_temporal + cond_BD_temporal).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
                if self.use_temporal_attn:
                    gamma_temporal, scale_temporal, shift_temporal = self.ada_lin_temporal(cond_BD_temporal).view(-1, 1, 3, self.C).unbind(2)

        if rope2d_freqs_grid is None:
            rope2d_freqs_grid_sa = None
            rope2d_freqs_grid_temporal = None
        elif isinstance(rope2d_freqs_grid, dict):
            rope2d_freqs_grid_sa = rope2d_freqs_grid[1]
            # rope2d_freqs_grid_sa = rope2d_freqs_grid[num_views]
            rope2d_freqs_grid_temporal = rope2d_freqs_grid[num_views]
        else:
            rope2d_freqs_grid_sa = rope2d_freqs_grid
            rope2d_freqs_grid_temporal = rope2d_freqs_grid
        
        if attn_bias_or_two_vector is not None:
            attn_bias_or_two_vector_sa = attn_bias_or_two_vector['sa']
            attn_bias_or_two_vector_temporal = attn_bias_or_two_vector['temporal']
        else:
            attn_bias_or_two_vector_sa = None
            attn_bias_or_two_vector_temporal = None

        if self.fused_norm_func is None:
            if self.use_local_attn:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                x_sa = self._sequence_to_scale_reshape(x_sa, num_views, scale_schedule)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector_sa, attn_fn, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, scale_ind=scale_ind, view_meta_data=None, timesteps=timesteps, num_views=1)
                    # x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, scale_ind=scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector_sa, attn_fn, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=None, timesteps=timesteps, num_views=1)
                    # x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                
                batch_size = x.shape[0]
                x_sa = self._scale_to_sequence_reshape(x_sa, num_views, batch_size, scale_schedule)
        
                x = x + self.drop_path(x_sa.mul_(gamma1))

            if self.use_temporal_attn:
                x_temporal = self.ln_wo_grad(x.float()).mul(scale_temporal.add(1)).add_(shift_temporal)

                view_meta_data_temporal = view_meta_data
                
                if self.checkpointing_sa_only and self.training:
                    x_temporal = checkpoint(self.sa_temporal, x_temporal, attn_bias_or_two_vector_temporal, attn_fn, scale_schedule, rope2d_freqs_grid_temporal, use_reentrant=False, scale_ind=scale_ind, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache)
                else:
                    x_temporal = self.sa_temporal(x_temporal, attn_bias_or_two_vector_temporal, attn_fn, scale_schedule, rope2d_freqs_grid_temporal, scale_ind=scale_ind, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache)

                if isinstance(x_temporal, tuple):
                    x_temporal, temporal_cache = x_temporal
                
                x = x + self.drop_path(x_temporal.mul_(gamma_temporal))
                
            x_ca = self.ca_norm(x)
            if x_ca.shape[0] != ca_kv[1].shape[0] - 1:
                # Separate process for each frame
                x_ca = self._sequence_to_scale_reshape(x_ca, num_views, scale_schedule)
            
            x_ca = self.ca(x_ca, ca_kv, scale_ind, scale_schedule, rope2d_freqs_grid_sa).float()
            
            if x.shape[0] != x_ca.shape[0]:
                x_ca = self._scale_to_sequence_reshape(x_ca, num_views, batch_size, scale_schedule)
            
            x = x + x_ca.mul_(self.ca_gamma)
            
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            raise NotImplementedError
        
        return x, temporal_cache
    
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'



def main():
    dev = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = torch.Generator(device=dev)
    # for Li in ([1, 3, 5], [1, 3]):
    rng.manual_seed(0)
    B, H, cq, ckv = 4, 8, 64, 96
    Cq = H*cq
    Ckv = H*ckv
    
    Li = [5, 4, 7, 6]
    Lq = 10
    L = max(Li)
    attn_bias = torch.zeros(B, 1, Lq, L, device=dev)
    for i, x in enumerate(Li):
        attn_bias[i, 0, :, x:] = -torch.inf
    
    q = torch.randn(B, Lq, H, cq, generator=rng, device=dev)
    k = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    v = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    tq, tk, tv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # BHLc
    
    seqlen_k = torch.tensor(Li, dtype=torch.int32, device=dev)
    cu_seqlens_k = F.pad(torch.cumsum(seqlen_k, dim=0, dtype=torch.torch.int32), (1, 0))
    kv = torch.stack([k, v], dim=2)
    kv_compact = torch.cat([kv[i, :Li[i]] for i in range(B)], dim=0)
    
    ca = CrossAttention(for_attn_pool=False, embed_dim=Cq, kv_dim=Ckv, num_heads=H)
    CrossAttention.forward
    ca(q, (kv_compact, cu_seqlens_k, max(Li))).mean().backward()


if __name__ == '__main__':
    main()
