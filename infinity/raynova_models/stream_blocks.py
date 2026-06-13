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


class RoPECrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False, use_condition_rope=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # TODO: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        self.use_condition_rope = use_condition_rope
            
    def forward(self, q, ca_kv, scale_ind, scale_schedule, rope2d_freqs_grid):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        if self.use_condition_rope:
            kv_compact, cu_seqlens_k, max_seqlen_k, center_compact = ca_kv
        else:
            kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens
        
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
    
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)

        if self.use_condition_rope:
            k, v = kv_compact.unbind(dim=1)
            q_compact, k = apply_rope_condition(q_compact, k, center_compact, rope2d_freqs_grid, scale_ind, scale_schedule, Lq)
            kv_compact = torch.stack((k, v), dim=1)

        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'



class MVSelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, use_flex_attn=False,
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
        return_cache=False, input_dim=None,
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
        self.return_cache = return_cache
        self.embed_dim = embed_dim
        
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
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool, caching_use_dict=False, prefix_cache=False, last_scale_only=False): # kv caching: only used during inference
        self.caching = enable
        self.caching_use_dict = caching_use_dict
        self.prefix_cache = prefix_cache
        self.last_scale_only = last_scale_only
        self.cached_k = None
        self.cached_v = None
        self.cached_k_cfg = None
        self.cached_v_cfg = None
    
    def clean_cfg_cache(self):
        self.cached_k_cfg = None
        self.cached_v_cfg = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=-1, view_meta_data=None, timesteps=None, num_views=None, past_cache=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector: block-wise, lower-triangle matrix where 0 means visible and -inf means invisible.
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, _ = x.shape
        C = self.embed_dim
        
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
        if self.return_cache:
            cached_k, cached_v = past_cache 
        else:
            cached_k, cached_v = self.cached_k, self.cached_v
        if self.caching:    # kv caching: only used during inference
            # cfg = x.shape[0] == 2
            cfg = False
            if cfg:
                k, k_cfg = k.split(1, dim=0)
                v, v_cfg = v.split(1, dim=0)
            if cached_k is None: 
                if self.caching_use_dict:
                    cached_k = {scale_ind: k.to(torch.bfloat16)}
                    cached_v = {scale_ind: v.to(torch.bfloat16)}
                elif self.last_scale_only:
                    if scale_ind == len(scale_schedule) - 1:
                        cached_k = k.clone().to(torch.bfloat16)
                        cached_v = v.clone().to(torch.bfloat16)
                else:
                    cached_k = k.clone().to(torch.bfloat16)
                    cached_v = v.clone().to(torch.bfloat16)
            else: 
                if self.caching_use_dict:
                    if scale_ind not in cached_k:
                        new_ck = k.to(torch.bfloat16) 
                        new_cv = v.to(torch.bfloat16)
                    else:
                        new_ck = torch.cat((cached_k[scale_ind], k.to(torch.bfloat16)), dim=L_dim)
                        new_cv = torch.cat((cached_v[scale_ind], v.to(torch.bfloat16)), dim=L_dim)
                    cached_k[scale_ind] = new_ck
                    cached_v[scale_ind] = new_cv
                    if self.prefix_cache:
                        k = []
                        v = []
                        for scale_ind_ in range(scale_ind+1):
                            k.append(cached_k[scale_ind_])
                            v.append(cached_v[scale_ind_])
                        k = torch.cat(k, dim=L_dim)
                        v = torch.cat(v, dim=L_dim)
                    else:
                        k = cached_k[scale_ind]
                        v = cached_v[scale_ind]
                elif self.last_scale_only:
                    if scale_ind == len(scale_schedule) - 1:
                        cached_k = torch.cat((cached_k, k), dim=L_dim)
                        cached_v = torch.cat((cached_v, v), dim=L_dim)
                        k = cached_k.clone()
                        v = cached_v.clone()
                    else:
                        k = torch.cat((cached_k, k), dim=L_dim)
                        v = torch.cat((cached_v, v), dim=L_dim)
                else:
                    k = torch.cat((cached_k, k), dim=L_dim)
                    cached_k = k.clone()
                    v = torch.cat((cached_v, v), dim=L_dim)
                    cached_v = v.clone()
            
            if cfg:
                if self.cached_k_cfg is None: 
                    if self.caching_use_dict:
                        self.cached_k_cfg = {scale_ind: k_cfg.to(torch.bfloat16)}
                        self.cached_v_cfg = {scale_ind: v_cfg.to(torch.bfloat16)}
                    else:
                        self.cached_k_cfg = k_cfg.clone().to(torch.bfloat16)
                        self.cached_v_cfg = v_cfg.clone().to(torch.bfloat16)
                else: 
                    if self.caching_use_dict:
                        if scale_ind not in self.cached_k_cfg:
                            new_ck = k_cfg.to(torch.bfloat16) 
                            new_cv = v_cfg.to(torch.bfloat16)
                        else:
                            new_ck = torch.cat((self.cached_k_cfg[scale_ind], k_cfg.to(torch.bfloat16)), dim=L_dim)
                            new_cv = torch.cat((self.cached_v_cfg[scale_ind], v_cfg.to(torch.bfloat16)), dim=L_dim)
                        self.cached_k_cfg[scale_ind] = new_ck
                        self.cached_v_cfg[scale_ind] = new_cv
                        if self.prefix_cache:
                            k_cfg = []
                            v_cfg = []
                            for scale_ind_ in range(scale_ind+1):
                                k_cfg.append(self.cached_k_cfg[scale_ind_])
                                v_cfg.append(self.cached_v_cfg[scale_ind_])
                            k_cfg = torch.cat(k_cfg, dim=L_dim)
                            v_cfg = torch.cat(v_cfg, dim=L_dim)
                        else:
                            k_cfg = cached_k[scale_ind]
                            v_cfg = cached_v[scale_ind]
                    else:
                        k_cfg = torch.cat((self.cached_k_cfg, k_cfg), dim=L_dim)
                        self.cached_k_cfg = k_cfg.clone()
                        v_cfg = torch.cat((self.cached_v_cfg, v_cfg), dim=L_dim)
                        self.cached_v_cfg = v_cfg.clone()

        if not self.return_cache:
            self.cached_k = cached_k
            self.cached_v = cached_v

        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            q = q.type_as(v)
            k = k.type_as(v)
            if self.caching and cfg:
                assert B == 2
                k_cfg = k_cfg.type_as(v)
                oup1 = slow_attn(query=q[:1], key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(1, L, C)
                oup2 = slow_attn(query=q[1:], key=k_cfg, value=v_cfg, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(1, L, C)
                oup = torch.cat([oup1, oup2], dim=0)
            else:
                oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)

        if self.return_cache:
            return self.proj_drop(self.proj(oup)), [cached_k, cached_v]
        else:
            return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'tau={self.tau}, cos_attn={self.cos_attn}'


class StreamAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        use_temporal_attn=True, use_condition_rope=False,
        use_local_attn=False,
    ):
        super(StreamAttnBlock, self).__init__()
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
            self.sa_temporal = MVSelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
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
        if past_cache is None:
            past_cache = {
                'sa': [None, None], 
                'temporal': [None, None]
            }
        cache_temporal = [None, None]
        cache_sa = [None, None]
        with torch.cuda.amp.autocast(enabled=False):    # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
                if not self.use_local_attn:
                    gamma_temporal, gamma2, scale_temporal, scale2, shift_temporal, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
                if self.use_temporal_attn:
                    gamma_temporal, scale_temporal, shift_temporal = (self.ada_gss_temporal + cond_BD_temporal).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
                if not self.use_local_attn:
                    gamma_temporal, gamma2, scale_temporal, scale2, shift_temporal, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
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
            # train
            if self.use_local_attn:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                x_sa = self._sequence_to_scale_reshape(x_sa, num_views, scale_schedule)
                if attn_fn is not None:
                    attn_fn_ = attn_fn[0][num_views]
                else: 
                    attn_fn_ = None
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector_sa, attn_fn_, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, scale_ind=scale_ind, view_meta_data=None, timesteps=timesteps, num_views=1, past_cache=past_cache['sa'])
                    # x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, scale_ind=scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector_sa, attn_fn_, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=None, timesteps=timesteps, num_views=1, past_cache=past_cache['sa'])
                    # x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                if isinstance(x_sa, tuple):
                    x_sa, cache_sa = x_sa
                
                batch_size = x.shape[0]
                x_sa = self._scale_to_sequence_reshape(x_sa, num_views, batch_size, scale_schedule)
        
                x = x + self.drop_path(x_sa.mul_(gamma1))
            
            if self.use_temporal_attn:
                x_temporal = self.ln_wo_grad(x.float()).mul(scale_temporal.add(1)).add_(shift_temporal)
                view_meta_data_temporal = view_meta_data
                if attn_fn is not None:
                    attn_fn_ = attn_fn[1][num_views]
                else:
                    attn_fn_ = None
                if self.checkpointing_sa_only and self.training:
                    x_temporal = checkpoint(self.sa_temporal, x_temporal, attn_bias_or_two_vector_temporal, attn_fn_, scale_schedule, rope2d_freqs_grid_temporal, use_reentrant=False, scale_ind=scale_ind, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache['temporal'])
                else:
                    x_temporal = self.sa_temporal(x_temporal, attn_bias_or_two_vector_temporal, attn_fn_, scale_schedule, rope2d_freqs_grid_temporal, scale_ind=scale_ind, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache['temporal'])

                if isinstance(x_temporal, tuple):
                    x_temporal, cache_temporal = x_temporal

                x = x + self.drop_path(x_temporal.mul_(gamma_temporal))
                
            x_ca = self.ca_norm(x)
            if x_ca.shape[0] != ca_kv[1].shape[0] - 1:
                if ca_kv[1].shape[0] - 1 == batch_size * timesteps:
                    x_ca = x_ca.reshape(batch_size*timesteps, -1, x_ca.shape[-1])
                else:
                    # Separate process for each frame
                    x_ca = self._sequence_to_scale_reshape(x_ca, num_views, scale_schedule)
            
            x_ca = self.ca(x_ca, ca_kv, scale_ind, scale_schedule, rope2d_freqs_grid_sa).float()
            
            if x.shape[0] != x_ca.shape[0]:
                if ca_kv[1].shape[0] - 1 == batch_size * timesteps:
                    x_ca = x_ca.reshape(batch_size, -1, x_ca.shape[-1])
                else:
                    x_ca = self._scale_to_sequence_reshape(x_ca, num_views, batch_size, scale_schedule)
            
            x = x + x_ca.mul_(self.ca_gamma)
            
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            # infer
            if self.use_local_attn:
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
                x_sa = self._sequence_to_scale_reshape(x_sa, num_views, scale_schedule)

                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector_sa, attn_fn, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, view_meta_data=None, timesteps=timesteps, num_views=1, past_cache=past_cache['sa'])
                    # x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, use_reentrant=False, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector_sa, attn_fn, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=None, timesteps=timesteps, num_views=1, past_cache=past_cache['sa'])
                    # x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid_sa, scale_ind=scale_ind, view_meta_data=view_meta_data, timesteps=timesteps, num_views=num_views, past_cache=past_cache['sa'])
                if isinstance(x_sa, tuple):
                    x_sa, cache_sa = x_sa

                batch_size = x.shape[0]
                x_sa = self._scale_to_sequence_reshape(x_sa, num_views, batch_size, scale_schedule)
                
                x = x + self.drop_path(x_sa.mul_(gamma1))
                

            if self.use_temporal_attn:
                x_temporal = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale_temporal, shift=shift_temporal)
                view_meta_data_temporal = view_meta_data

                if self.checkpointing_sa_only and self.training:
                    x_temporal = checkpoint(self.sa_temporal, x_temporal, attn_bias_or_two_vector_temporal, attn_fn, scale_schedule, rope2d_freqs_grid_temporal, use_reentrant=False, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache['temporal'])
                else:
                    x_temporal = self.sa_temporal(x_temporal, attn_bias_or_two_vector_temporal, attn_fn, scale_schedule, rope2d_freqs_grid_temporal, scale_ind=scale_ind, view_meta_data=view_meta_data_temporal, timesteps=timesteps, num_views=num_views, past_cache=past_cache['temporal'])
                if isinstance(x_temporal, tuple):
                    x_temporal, cache_temporal = x_temporal

                x = x + self.drop_path(x_temporal.mul_(gamma_temporal))
                
            x_ca = self.ca_norm(x)
            if x_ca.shape[0] != ca_kv[1].shape[0] - 1:
                if ca_kv[1].shape[0] - 1 == batch_size * timesteps:
                    x_ca = x_ca.reshape(batch_size*timesteps, -1, x_ca.shape[-1])
                else:
                    # Separate process for each frame
                    x_ca = self._sequence_to_scale_reshape(x_ca, num_views, scale_schedule)
            
            x_ca = self.ca(x_ca, ca_kv, scale_ind, scale_schedule, rope2d_freqs_grid_sa).float()
            
            if x.shape[0] != x_ca.shape[0]:
                if ca_kv[1].shape[0] - 1 == batch_size * timesteps:
                    x_ca = x_ca.reshape(batch_size, -1, x_ca.shape[-1])
                else:
                    x_ca = self._scale_to_sequence_reshape(x_ca, num_views, batch_size, scale_schedule)

            x = x + x_ca.mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP

        return x
    
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
