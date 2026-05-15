"""
Definition of Infinity transformer model.
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from tkinter import NONE
from typing import List, Optional, Tuple, Union, Dict, Any
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid
from infinity.raynova_models.stream_blocks import StreamAttnBlock
from infinity.raynova_models.recurrent_blocks import RecurrentAttnBlock

from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from infinity.models.embedder import get_embedder
from infinity.raynova_models.box_embedder import ContinuousBBoxWithTextEmbedding
from infinity.raynova_models.map_embedder import MapWithTextEmbedding
from infinity.utils.misc import get_scene_description, project_corners_to_views, remove_cache
from infinity.models.infinity import MultiInpIdentity, TextAttentivePool

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None
    
from scenarionet_tools.map_utils import convert_points

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class SharedAdaLin(nn.Linear):  
    def __init__(self, in_features, out_features, oup_num=6):
        super().__init__(in_features, out_features)
        self.oup_num = oup_num
    def forward(self, cond_BD):
        C = self.weight.shape[0] // self.oup_num
        return super().forward(cond_BD).reshape(cond_BD.shape[0], -1, self.oup_num, C)   # B, 1, oup_num, C

class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])
            

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, timesteps=None, num_views=None, view_meta_data=None, scale_ind=-1, past_cache=None, cond_BD_temporal=None):
        h = x
        new_cache = []
        if past_cache is None:
            past_cache = [None] * len(self.module)
        for mid, m in enumerate(self.module):
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind, num_views, timesteps, view_meta_data, cond_BD_temporal, past_cache[mid], use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind, num_views, timesteps, view_meta_data, cond_BD_temporal, past_cache[mid])
            if isinstance(h, tuple):
                h, cache_module = h
                new_cache.append(cache_module)
        if len(new_cache) > 0:
            return h, new_cache
        else:
            return h

class RAYNOVA(nn.Module):
    def __init__(
        self, vae_local,
        text_channels=2048, text_maxlen=512,     # text-cond generation
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4.,   # model's architecture
        drop_rate=0., drop_path_rate=0.,    # drop out and drop path
        norm_eps=1e-6, rms_norm=False,      # norm layer
        shared_aln=False, head_aln=True,    # adaptive norm
        cond_drop_rate=0.1,                 # for classifier-free guidance
        rand_uncond=False,
        cross_attn_layer_scale=-1., nm0=False, tau=1, cos_attn=True, swiglu=False,
        raw_scale_schedule=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        head_depth=1,
        top_p=0.0, top_k=0.0,
        fused_mlp=False, fused_norm=False,
        block_chunks=1,
        checkpointing=None,
        pad_to_multiplier=0,
        use_flex_attn=False,
        batch_size=2,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=0,
        rope2d_normalized_by_hw=0,
        pn=None,
        train_h_div_w_list=None,
        video_frames=1,
        always_training_scales=20,
        apply_spatial_patchify = 0,
        inference_mode=False,
        # for multiview
        num_views=6,
        timesteps=2,
        view_embed_type='ray',
        attn_bias_type='full',
        add_view_embeding_only_first_block=0,
        add_time_embeding_only_first_block=1,
        object_condition=0,
        class_names=[],
        object_cond_drop_rate=0.25,
        bbox_img_coord=0,
        time_embed_type='fourier',
        max_ray_depth=50, ray_normalize=[50, 50, 10],
        recurrent_training=0,
        use_temporal_attn=1,
        use_condition_rope=0,
        map_condition=0,
        map_names=[],
        map_condition_drop_rate=0.25,
        map_sample_points_num=20,
        use_frame_coordinate=0,
        freeze_backbone=False,
        time_chunk=None,
        use_local_attn=True,
        adaptive_horizon=0,
        max_horizon=16,
    ):
        # set hyperparameters
        self.C = embed_dim
        self.inference_mode = inference_mode
        self.apply_spatial_patchify = apply_spatial_patchify
        if self.apply_spatial_patchify:
            self.d_vae = vae_local.embed_dim * 4
        else:
            self.d_vae = vae_local.embed_dim
        self.use_bit_label = use_bit_label
        self.codebook_dim = self.d_vae
        self.V = (self.codebook_dim * 2) if self.use_bit_label else vae_local.vocab_size
        self.bit_mask = vae_local.quantizer.lfq.mask if self.use_bit_label else None
        self.Ct5 = text_channels
        self.depth = depth
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.mlp_ratio = mlp_ratio
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = norm_eps
        self.prog_si = -1
        self.pn = pn
        self.train_h_div_w_list = []
        self.train_h_div_w_list = train_h_div_w_list if train_h_div_w_list else h_div_w_templates
        self.train_h_div_w_list = [0.571]
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales

        self.use_local_attn = use_local_attn
        self.adaptive_horizon = adaptive_horizon
        self.max_horizon = max_horizon
        
        assert add_lvl_embeding_only_first_block in [0,1]
        self.add_lvl_embeding_only_first_block = add_lvl_embeding_only_first_block
        assert rope2d_each_sa_layer in [0,1]
        self.rope2d_each_sa_layer = rope2d_each_sa_layer
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        print(f'self.codebook_dim: {self.codebook_dim}, self.add_lvl_embeding_only_first_block: {self.add_lvl_embeding_only_first_block}, \
            self.use_bit_label: {self.use_bit_label}, self.rope2d_each_sa_layer: {rope2d_each_sa_layer}, self.rope2d_normalized_by_hw: {self.rope2d_normalized_by_hw}')
        head_up_method = ''
        word_patch_size = 1 if head_up_method in {'', 'no'} else 2
        if word_patch_size > 1:
            assert all(raw_pn % word_patch_size == 0 for raw_pn in raw_scale_schedule), f'raw_scale_schedule={raw_scale_schedule}, not compatible with word_patch_size={word_patch_size}'
        
        self.checkpointing = checkpointing
        self.pad_to_multiplier = max(1, pad_to_multiplier)

        self.raw_scale_schedule = raw_scale_schedule    # 'raw' means before any patchifying
        self.first_l = num_views
        # solve top-p top-k sampling hyperparameters
        self.top_p, self.top_k = max(min(top_p, 1), 0), (round(top_k * self.V) if 0 < top_k < 1 else round(top_k))
        if self.top_p < 1e-5: self.top_p = 0
        if self.top_k >= self.V or self.top_k <= 0: self.top_k = 0
        
        t = torch.zeros(dist.get_world_size(), device=dist.get_device())
        t[dist.get_rank()] = float(flash_fused_op_installed)
        dist.barrier()
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'flash_fused_op_installed: {t}'
        
        super().__init__()
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.text_maxlen = text_maxlen

        self.num_views = num_views
        self.timesteps = timesteps
        self.add_view_embeding_only_first_block = add_view_embeding_only_first_block
        assert add_view_embeding_only_first_block in [0, 1]
        self.view_embed_type = view_embed_type
        self.attn_bias_type = attn_bias_type
        print(f'self.view_embed_type: {self.view_embed_type}, self.add_view_embeding_only_first_block: {self.add_view_embeding_only_first_block}')
        print(f"self.top_p: {self.top_p}", f"self.top_k: {self.top_k}")
        if 'ray' in view_embed_type:
            self.max_ray_depth = max_ray_depth
            self.ray_normalize = ray_normalize
            if 'plucker' in view_embed_type:
                self.view_embed = nn.Sequential(
                    nn.Linear(6, self.C),
                    nn.GELU(approximate='tanh'), # approximately added on Apr. 19
                    zero_module(nn.Linear(self.C, self.C)),
                )
            else:
                self.view_embed = nn.Sequential(
                    nn.Linear(self.max_ray_depth*3, self.C),
                    nn.GELU(approximate='tanh'), # approximately added on Apr. 19
                    zero_module(nn.Linear(self.C, self.C)),
                )
        elif view_embed_type == "none":
            pass
        else:
            raise NotImplementedError
        
        self.recurrent_training = recurrent_training
        assert recurrent_training in [0, 1]

        self.use_temporal_attn = use_temporal_attn

        self.object_condition = object_condition
        self.use_condition_rope = use_condition_rope
        self.class_names = class_names
        self.object_cond_drop_rate = object_cond_drop_rate
        self.map_condition = map_condition
        self.map_condition_drop_rate = map_condition_drop_rate
        self.map_sample_points_num = map_sample_points_num
        self.use_frame_coordinate = use_frame_coordinate
        if object_condition:
            self.bbox_img_coord = bbox_img_coord
            if bbox_img_coord:
                mode = 'all-xyz-6d'
            else:
                mode = 'all-xyz'

            self.bbox_encoder = ContinuousBBoxWithTextEmbedding(
                n_classes=len(class_names), output_dim=self.C, class_token_dim=self.Ct5, trainable_class_token=False,
                embedder_num_freq=8, proj_dims=[768, 768], use_text_encoder_init=True, mode=mode
            )
            self.object_norm = FastRMSNorm(self.C, elementwise_affine=True, eps=norm_eps)
            self.object_proj_for_ca = nn.Sequential(
                nn.Linear(self.C, self.C),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.C, self.C),
            )

        if map_condition:
            if bbox_img_coord:
                mode = 'all-xyz-6d'
            else:
                mode = 'all-xyz'
            self.map_encoder = MapWithTextEmbedding(
                n_classes=len(map_names), output_dim=self.C, class_token_dim=self.Ct5, trainable_class_token=False,
                proj_dims=[64, 64, 128, 768, 512], use_text_encoder_init=True, mode=mode, minmax_normalize=True,
                center_normalize=True
            )
            self.map_norm = FastRMSNorm(self.C, elementwise_affine=True, eps=norm_eps)
            self.map_proj_for_ca = nn.Sequential(
                nn.Linear(self.C, self.C),
                nn.GELU(approximate='tanh'),
                nn.Linear(self.C, self.C),
            )


        #TODO: a neater version of time embedding
        self.time_embed_type = time_embed_type
        self.add_time_embeding_only_first_block = add_time_embeding_only_first_block
        assert time_embed_type in ['fourier', 'none']
        if time_embed_type == 'fourier':
            self.time_fourier_embedder = get_embedder(1, 8)
            self.time_embedder = nn.Sequential(
                nn.Linear(16, 256),
                nn.GELU(approximate='tanh'),
                zero_module(nn.Linear(256, self.C))
            )
            
        # [inp & position embedding]
        init_std = math.sqrt(1 / self.C / 3)
        self.norm0_cond = nn.Identity()
        self.D = self.C

        cfg_uncond = torch.empty(self.text_maxlen, self.Ct5)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.Ct5 ** 0.5
        if rand_uncond:
            self.register_buffer('cfg_uncond', cfg_uncond)
        else:
            self.cfg_uncond = nn.Parameter(cfg_uncond)

        self.text_norm = FastRMSNorm(self.Ct5, elementwise_affine=True, eps=norm_eps)
        self.text_proj_for_sos = TextAttentivePool(self.Ct5, self.D)
        self.text_proj_for_ca = nn.Sequential(
            nn.Linear(self.Ct5, self.D),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.D, self.D),
        )

        #NOTE: the first tokens of 6 views share the same start?
        self.pos_1LsC_start = nn.Parameter(torch.empty(1, self.first_l//num_views, self.C)) # 1, 1, C
        nn.init.trunc_normal_(self.pos_1LsC_start.data, mean=0, std=init_std)

        if self.rope2d_each_sa_layer or 'prope' in self.view_embed_type:
            rope2d_freqs_grid = {}
            for nv in range(1, self.num_views+1):
                rope2d_freqs_grid_view = precompute_rope2d_freqs_grid(dim=self.C//self.num_heads, dynamic_resolution_h_w=dynamic_resolution_h_w, pad_to_multiplier=self.pad_to_multiplier, rope2d_normalized_by_hw=self.rope2d_normalized_by_hw, num_views=nv, freeze_backbone=freeze_backbone)
                rope2d_freqs_grid[nv] = rope2d_freqs_grid_view
            self.rope2d_freqs_grid = rope2d_freqs_grid
        else:
            self.rope2d_freqs_grid = None
            # raise ValueError(f'self.rope2d_each_sa_layer={self.rope2d_each_sa_layer} not implemented')
        self.lvl_embed = nn.Embedding(15, self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # [input layers] input norm && input embedding
        norm_layer = partial(FastRMSNorm if rms_norm else nn.LayerNorm, eps=norm_eps)
        self.norm0_ve = norm_layer(self.d_vae) if nm0 else nn.Identity()
        self.word_embed = nn.Linear(self.d_vae, self.C)
        
        # [shared adaptive layernorm mapping network]
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        if self.use_temporal_attn:
            self.shared_ada_lin_temporal = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 3*self.C, 3)) if shared_aln else nn.Identity()

        # fused norm
        if fused_norm:
            fused_norm_func = fused_ada_rms_norm if rms_norm else fused_ada_layer_norm
            if fused_norm_func is not None: # pre-compile
                B = 2
                x = torch.randn(B, 1, self.C).requires_grad_(True)
                scale = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                shift = torch.randn(B, 1, self.C).mul_(0.01).requires_grad_(True)
                # fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale, shift=shift).mean().backward()
                del B, x, scale, shift
        else:
            fused_norm_func = None
        
        # [backbone and head]
        self.use_flex_attn = use_flex_attn
        self.attn_fn_compile_dict = {}
        self.batch_size = batch_size
        if self.use_flex_attn:
            self.attn_fn_compile_dict = self.compile_flex_attn()

        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # dpr means drop path rate (linearly increasing)
        self.unregistered_blocks = []
        for block_idx in range(depth):
            if self.recurrent_training:
                block = RecurrentAttnBlock(
                    embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                    swiglu=swiglu, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                    checkpointing_sa_only=self.checkpointing == 'self-attn',
                    use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                    use_temporal_attn=use_temporal_attn, use_condition_rope=use_condition_rope,
                    time_chunk=time_chunk, use_local_attn=use_local_attn
                )
            else:
                block = StreamAttnBlock(
                    embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=cross_attn_layer_scale, cond_dim=self.D, act=True, shared_aln=shared_aln, norm_layer=norm_layer,
                    num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[block_idx], tau=tau, cos_attn=cos_attn,
                    swiglu=swiglu, fused_mlp=fused_mlp, fused_norm_func=fused_norm_func,
                    checkpointing_sa_only=self.checkpointing == 'self-attn',
                    use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                    use_temporal_attn=use_temporal_attn, use_condition_rope=use_condition_rope,
                    use_local_attn=use_local_attn
                )
            self.unregistered_blocks.append(block)
        
        # [head]
        V = self.V
        if head_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, act=True, norm_layer=norm_layer, fused_norm_func=fused_norm_func)
            self.head = nn.Linear(self.C, V) if head_depth == 1 else nn.Sequential(nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, V)) if head_depth == 1 else nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.C, bias=True), nn.GELU(approximate='tanh'), nn.Linear(self.C, V))
        
        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // block_chunks
        print(f"{self.num_blocks_in_a_chunk=}, {depth=}, {block_chunks=}")
        assert self.num_blocks_in_a_chunk * block_chunks == depth
        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i*self.num_blocks_in_a_chunk))

    def _flatten_object_condition(self, object_condition, scale_schedule, need_to_pad=0, num_views=None):
        # object_condition: [B, T, V, C]
        if num_views is None:
            num_views = self.num_views
        object_condition_LC = []
        for i, (pt, ph, pw) in enumerate(scale_schedule):
            for vid in range(num_views):
                object_condition_LC.append(object_condition[:, :, vid:vid+1].repeat(1, 1, pt*ph*pw, 1))
        object_condition_LC = torch.cat(object_condition_LC, dim=2).flatten(1, 2)  # [B, L, C]
        object_condition_LC = torch.cat([object_condition_LC, torch.zeros_like(object_condition_LC[:, :need_to_pad])], dim=1)
        return object_condition_LC


    def _generate_fourier_time_embedding(self, meta_data_list, L_per_step, need_to_pad=0, timesteps=None):
        pos_time_BL = []
        if timesteps is None:
            timesteps = self.timesteps
        for tid in range(timesteps):
            batch_timestep = meta_data_list[tid]['timestep'][:, None]  # [B,]
            batch_timestep = batch_timestep.repeat(1, L_per_step)
            pos_time_BL.append(batch_timestep)
        
        pos_time_BL = torch.cat(pos_time_BL, dim=1)
        pos_time_BLC = self.time_embedder(self.time_fourier_embedder(pos_time_BL[...,None]))
        return pos_time_BLC

    def compile_flex_attn(self):
        attn_fn_compile_dict = {}
        for h_div_w in self.train_h_div_w_list:
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][self.pn]['scales']
            if self.inference_mode:
                apply_flex_attn_scales = list(range(1, 1+len(full_scale_schedule)))
                mask_type = "var_infer_mask_with_kv_cache"
                auto_padding = True
            else:
                mask_type = 'var'
                auto_padding = True
                apply_flex_attn_scales = [min(self.always_training_scales, len(full_scale_schedule))]
            for scales_num in apply_flex_attn_scales:
                print(f'====== apply flex attn hdivw: {h_div_w} scales: {scales_num} ======')
                scale_schedule = full_scale_schedule[:scales_num]
                scale_schedule = [ (min(t, self.video_frames//4+1), h, w) for (t,h, w) in scale_schedule]

                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L

                attn_fn_temporal_list = {}
                attn_fn_list = {}
                for nv in range(1, self.num_views+1):
                    if self.adaptive_horizon:
                        num_frames = self.num_views * self.timesteps // nv
                        num_frames = min(num_frames, self.max_horizon)
                    else:
                        num_frames = self.timesteps
                    attn_fn_sa_v = FlexAttn(block_scales = patchs_nums_tuple,
                                            mask_type = mask_type,
                                            B = self.batch_size*nv*num_frames, 
                                            H = self.num_heads,
                                            L = aligned_L,
                                            auto_padding=auto_padding)
                    attn_fn_temporal_v = FlexAttn(block_scales = patchs_nums_tuple,
                                            mask_type = self.attn_bias_type,
                                            B = self.batch_size, 
                                            H = self.num_heads,
                                            num_views = nv,
                                            timesteps = num_frames,
                                            L = aligned_L*nv*num_frames,
                                            auto_padding=auto_padding)
                    attn_fn_list[nv] = attn_fn_sa_v
                    attn_fn_temporal_list[nv] = attn_fn_temporal_v
                attn_fn_compile_dict[tuple(scale_schedule)] = (attn_fn_list, attn_fn_temporal_list)

            if self.video_frames > 1: # append image attn_fn when self.video_frames > 1 (namely videos)
                raise NotImplementedError
                scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
                patchs_nums_tuple = tuple(scale_schedule)
                SEQ_L = sum( pt * ph * pw * self.num_views for pt, ph, pw in patchs_nums_tuple)
                aligned_L = SEQ_L+ (self.pad_to_multiplier - SEQ_L % self.pad_to_multiplier) if SEQ_L % self.pad_to_multiplier != 0 else SEQ_L
                attn_fn = FlexAttn(block_scales = patchs_nums_tuple,
                                        mask_type = mask_type,
                                        B = self.batch_size, 
                                        H = self.num_heads,
                                        L = aligned_L)
                attn_fn_compile_dict[patchs_nums_tuple] = attn_fn
        return attn_fn_compile_dict
        
    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        """
        :param h: hidden_state, shaped (B or batch_size, L or seq_len, C or hidden_dim)
        :param cond_BD: shaped (B or batch_size, D or cond_dim)
        :param tau: temperature
        :return: logits, shaped (B or batch_size, V or vocabulary_size)
        """
        with torch.amp.autocast('cuda', enabled=False):
            return self.head(self.head_nm(h.float(), cond_BD.float()))

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0, num_views=None):
        if num_views is None:
            num_views = self.num_views
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w * num_views + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w*num_views] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w * num_views),dtype=torch.int).to(feature.device))
        return feature
    
    def add_lvl_embeding_for_x_BLC(self, x_BLC, scale_schedule, need_to_pad=0, num_views=None, timesteps=None):
        if num_views is None:
            num_views = self.num_views
        if self.recurrent_training:
            timesteps = 1
        else:
            if timesteps is None:
                timesteps = self.timesteps
        ptr = 0
        x_BLC_list = []
        for tid in range(timesteps):
            for scale_ind, patch_t_h_w in enumerate(scale_schedule):
                scale_seq_len = np.array(patch_t_h_w).prod() * num_views
                x_BLC_this_scale = x_BLC[:,ptr:ptr+scale_seq_len] # shape: [bs, patch_h*patch_w*num_views, c]
                ptr += scale_seq_len
                x_BLC_this_scale = self.add_lvl_embeding(x_BLC_this_scale, scale_ind, scale_schedule, num_views=num_views)
                x_BLC_list.append(x_BLC_this_scale)
        assert x_BLC.shape[1] == (ptr + need_to_pad), f'{x_BLC.shape[1]} != {ptr} + {need_to_pad}'
        x_BLC_list.append(x_BLC[:,ptr:])
        x_BLC = torch.cat(x_BLC_list, dim=1)
        return x_BLC

    def _get_separate_attn_bias(self, attn_bias_for_masking, scale_schedule):
        view_ids = []
        for i, (pt, ph, pw) in enumerate(scale_schedule):
            for vid in range(self.num_views):
                view_ids.extend([vid]*ph*pw)    
        view_ids = torch.Tensor(view_ids)

        view_ids_mask = view_ids[:, None] == view_ids[None]
        view_bias = torch.where(view_ids_mask, 0, -torch.inf)
        view_bias = view_bias.reshape(1, 1, view_bias.shape[-2], view_bias.shape[-1])

        attn_bias_for_masking = torch.minimum(view_bias, attn_bias_for_masking)
        return attn_bias_for_masking


    def _get_independent_attn_bias(self, attn_bias_for_masking, scale_schedule, num_views, L_per_step, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        view_ids = []
        for i, (pt, ph, pw) in enumerate(scale_schedule):
            for vid in range(num_views):
                view_ids.extend([vid]*ph*pw)    
        view_ids = torch.Tensor(view_ids)
        view_ids = torch.cat([view_ids]*timesteps, dim=0)

        view_ids_mask = view_ids[:, None] == view_ids[None]
        view_bias = torch.where(view_ids_mask, 0, -torch.inf)
        view_bias = view_bias.reshape(1, 1, view_bias.shape[-2], view_bias.shape[-1])

        attn_bias_for_masking = torch.minimum(view_bias, attn_bias_for_masking)
        
        for i in range(timesteps):
            for j in range(timesteps):
                if j != i:
                    attn_bias_for_masking[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf

        return attn_bias_for_masking


    def _attn_to_all_previous(self, attn_mask: torch.Tensor, L_per_step : int, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        
        for i in range(timesteps):
            for j in range(timesteps):
                if j > i:
                    attn_mask[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
                elif j < i:
                    attn_mask[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = 0.
            
        return attn_mask
    
    def _attn_mask_markov(self, attn_mask_same_scale: torch.Tensor, attn_mask_last_scale: torch.Tensor, L_per_step : int, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        for i in range(timesteps):
            for j in range(timesteps):
                if j > i:
                    attn_mask_last_scale[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
                    attn_mask_same_scale[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
                elif j == i:
                    attn_mask_last_scale[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
                elif j < i:
                    attn_mask_same_scale[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
        
        attn_mask = torch.maximum(attn_mask_same_scale, attn_mask_last_scale)
        return attn_mask
    

    def _attn_to_all_previous_low_scale(self, attn_mask_: torch.Tensor, L_per_step : int, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        attn_mask = torch.zeros_like(attn_mask_)
        for i in range(timesteps):
            for j in range(timesteps):
                if j > i:
                    attn_mask[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
                else:
                    attn_mask[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = 0.
        attn_mask = torch.minimum(attn_mask, attn_mask_)
        return attn_mask


    def _attn_to_self(self, attn_mask: torch.Tensor, L_per_step : int, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        for i in range(timesteps):
            for j in range(timesteps):
                if j != i:
                    attn_mask[:, :, i*L_per_step:(i+1)*L_per_step, j*L_per_step:(j+1)*L_per_step] = -torch.inf
            
        return attn_mask
    
    def _generate_rays_multi(self, meta_data_list, scale_schedule, local=False, plucker=False):
        ray_pos_list = []
        trans_matrix = torch.eye(4).type_as(meta_data_list[0]['curr_to_prev_lidar'])[None].repeat(meta_data_list[0]['curr_to_prev_lidar'].shape[0], 1, 1)
        for tid, meta_data in enumerate(meta_data_list):
            # Generate rays for each frame
            ray_pos = self._generate_rays(meta_data, scale_schedule, normalize=False)
            ray_pos = ray_pos.reshape(ray_pos.shape[0], ray_pos.shape[1], self.max_ray_depth, 3)
            ray_pos_pad = torch.cat([ray_pos, torch.ones_like(ray_pos[..., :1])], dim=-1)
            
            # Include ego-motion
            if not local:
                if 'curr_to_first_lidar' in meta_data_list[tid]:
                    trans_matrix = meta_data_list[tid]['curr_to_first_lidar']
                elif tid != 0:
                    trans_matrix = trans_matrix @ meta_data_list[tid]['curr_to_prev_lidar']

            ray_pos_pad = trans_matrix[:, None, None] @ ray_pos_pad[..., None]
            ray_pos = ray_pos_pad.squeeze(-1)[..., :3]
            ray_pos_list.append(ray_pos)

        ray_pos = torch.cat(ray_pos_list, dim=1)
        if plucker:
            cam_pos = ray_pos[:, :, 0]
            ray_dir = ray_pos[:, :, 10] - cam_pos
            ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
            ray_m = torch.cross(ray_dir, cam_pos, dim=-1)
            ray_pos = torch.cat([ray_m, ray_dir], dim=-1)
        else:
            # Normalize roughly not strictly to (-1, 1)
            for i in range(3):
                ray_pos[..., i] = ray_pos[..., i] / self.ray_normalize[i]
            ray_pos = ray_pos.flatten(2, 3)
        return ray_pos

    def _generate_rays(self, meta_data, scale_schedule, normalize=True):
        B, V = meta_data["rot"].shape[:2]
        device = meta_data["rot"].device
        width, height = meta_data["size"]
        
        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x V x 3 or B x V x 3 x 3
        cam_to_cam_aug = meta_data['rot'].new_zeros((B, V, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = meta_data['post_rot']
        cam_to_cam_aug[:, :, :3, 3] = meta_data['post_trans']

        intrins4x4 = meta_data['rot'].new_zeros((B, V, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = meta_data['intrins']

        cam_to_lidar_aug = meta_data['rot'].new_zeros((B, V, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = meta_data['rot']
        cam_to_lidar_aug[:, :, :3, 3] = meta_data['trans']

        ## Okay, to go from global to augmed cam (XYD)....
        # We can go from global to (X*D, X*D, D), then we need user to divide by D,
        # Then we can apply the img augs. So, we need to store both.
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug

        lidar_to_img = cam_to_cam_aug @ intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
        lidar_to_img = lidar_to_img.to(torch.float32)
        # Currently it cannot support 3D space augmentation

        ray_pos = []
        ar = width / height
        for lvl_id, (pt, ph, pw) in enumerate(scale_schedule):
            xs = torch.arange(pw).to(device) + 0.5
            ys = torch.arange(ph).to(device) + 0.5
            
            xs = xs * width / pw
            ys = ys * height / ph

            index  = torch.arange(start=0, end=self.max_ray_depth, step=1, device=device)
            index_1 = index + 1
            bin_size = self.max_ray_depth / (self.max_ray_depth * (1 + self.max_ray_depth))
            ds = bin_size * index * index_1

            xs = xs[None, :, None].repeat(ph, 1, self.max_ray_depth)
            ys = ys[:, None, None].repeat(1, pw, self.max_ray_depth)
            ds = ds[None, None, :].repeat(ph, pw, 1)

            xs = xs * ds
            ys = ys * ds
            img_coords_3d = torch.stack([xs, ys, ds], dim=-1)

            img_coords_4d = torch.cat([img_coords_3d, torch.ones_like(img_coords_3d[..., :1])], dim=-1)
            lidar_coords_4d = torch.matmul(torch.inverse(lidar_to_img)[:, :, None, None, None], img_coords_4d[None, None, ..., None]).squeeze(-1)
            lidar_coords_3d = lidar_coords_4d[..., :3]
    
            lidar_coords_3d = lidar_coords_3d.flatten(1,3)
            if normalize:
                for i in range(3):
                    lidar_coords_3d[..., i] = lidar_coords_3d[..., i] / self.ray_normalize[i]
                
            lidar_coords_3d = lidar_coords_3d.flatten(2,3)
            ray_pos.append(lidar_coords_3d)
            
        ray_pos = torch.cat(ray_pos, dim=1)
        return ray_pos # B x sum(1*1*v + 2*2*v + ...) x D3

    def _merge_text_bbox_condition(self, ca_kv, bbox_ca_kv, B, num_views=None, timesteps=None):
        text_kv_compact, text_cu_seqlens_k, _ = ca_kv
        if bbox_ca_kv is not None:
            if self.use_condition_rope:
                bbox_kv_compact, bbox_cu_seqlens_k, bbox_center_compact = bbox_ca_kv
            else:
                bbox_kv_compact, bbox_cu_seqlens_k = bbox_ca_kv
        
        merged_kv_compact = []
        if self.use_condition_rope:
            merged_center_compact = []
        merged_cu_seqlens_k = [torch.tensor(0).type_as(text_cu_seqlens_k[0])]
        merged_max_seqlen_k = 0
        
        sequence_text = (B == len(text_cu_seqlens_k) - 1)
        if self.training and not self.recurrent_training:
            if timesteps is None:
                timesteps = self.timesteps
        else:
            timesteps = 1
        
        if num_views is None:
            num_views = (len(bbox_cu_seqlens_k) - 1) // (timesteps * B)
            assert (len(bbox_cu_seqlens_k) - 1) % (timesteps * B) == 0
        for bid in range(B):
            for tid in range(timesteps):
                for vid in range(num_views):
                    sample_id = tid * (B * num_views) + vid * B + bid
                    if sequence_text:
                        text_sid = text_cu_seqlens_k[bid]
                        text_eid = text_cu_seqlens_k[bid+1]
                    else:
                        text_id = bid * timesteps + tid
                        text_sid = text_cu_seqlens_k[text_id]
                        text_eid = text_cu_seqlens_k[text_id+1]
                    text_kv_compact_item = text_kv_compact[text_sid:text_eid]
                    merged_kv_compact.append(text_kv_compact_item)

                    if bbox_ca_kv is not None:
                        bbox_sid = bbox_cu_seqlens_k[sample_id]
                        bbox_eid = bbox_cu_seqlens_k[sample_id+1]
                        box_kv_compact_item = bbox_kv_compact[bbox_sid:bbox_eid]
                    
                        merged_kv_compact.append(box_kv_compact_item)
                    else:
                        bbox_eid = 0
                        bbox_sid = 0
                
                    if self.use_condition_rope:
                        text_center_compact_item = torch.Tensor([[0.5, 0.5, 1]]).type_as(text_kv_compact)
                        text_center_compact_item = text_center_compact_item.repeat(text_eid-text_sid, 1)
                        box_center_compact_item = bbox_center_compact[bbox_sid:bbox_eid]
                        merged_center_compact.append(text_center_compact_item)
                        merged_center_compact.append(box_center_compact_item)
                
                    len_item = text_eid - text_sid + bbox_eid - bbox_sid
                    merged_cu_seqlens_k.append(merged_cu_seqlens_k[-1]+len_item)
                    merged_max_seqlen_k = max(merged_max_seqlen_k, len_item)

        merged_kv_compact = torch.cat(merged_kv_compact, dim=0)
        merged_cu_seqlens_k = torch.stack(merged_cu_seqlens_k)
        if self.use_condition_rope:
            merged_center_compact = torch.cat(merged_center_compact, dim=0)
        
            return (merged_kv_compact, merged_cu_seqlens_k, merged_max_seqlen_k, merged_center_compact)
        else:
            return (merged_kv_compact, merged_cu_seqlens_k, merged_max_seqlen_k)

    def _get_camera_parameters(self, meta_data):
        B, V = meta_data["rot"].shape[:2]
        device = meta_data["rot"].device
        width, height = meta_data["size"]
        
        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x V x 3 or B x V x 3 x 3
        cam_to_cam_aug = meta_data['rot'].new_zeros((B, V, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = meta_data['post_rot']
        cam_to_cam_aug[:, :, :3, 3] = meta_data['post_trans']

        intrins4x4 = meta_data['rot'].new_zeros((B, V, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = meta_data['intrins']
        
        cam_intrins = cam_to_cam_aug @ intrins4x4

        cam_to_lidar_aug = meta_data['rot'].new_zeros((B, V, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = meta_data['rot']
        cam_to_lidar_aug[:, :, :3, 3] = meta_data['trans']

        cam_extrins =  torch.inverse(cam_to_lidar_aug) 
        
        cam_parameter = {
            "intrinsic": cam_intrins,
            "extrinsic": cam_extrins,
            "extrinsic_inv": cam_to_lidar_aug,
        }
        return cam_parameter


    def _get_view_meta_data(self, cam_parameter, img_size, scale_schedule):
        B, V = cam_parameter['intrinsic'].shape[:2]
        device = cam_parameter['intrinsic'].device
        frame_intrinsic = []
        frame_extrinsic = []
        frame_extrinsic_inv = []
        frame_img_coords = []
        img_w, img_h = img_size
        for lvl_id, (pt, ph, pw) in enumerate(scale_schedule):
            lvl_intrinsic = cam_parameter['intrinsic'][:, :, None].repeat(1, 1, ph*pw, 1, 1)  # [B, V, L_sf_sv_ss, 4, 4]
            lvl_extrinsic = cam_parameter['extrinsic'][:, :, None].repeat(1, 1, ph*pw, 1, 1)  # [B, V, L_sf_sv_ss, 4, 4]
            lvl_extrinsic_inv = cam_parameter['extrinsic_inv'][:, :, None].repeat(1, 1, ph*pw, 1, 1)  # [B, V, L_sf_sv_ss, 4, 4]
            lvl_intrinsic = lvl_intrinsic.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
            lvl_extrinsic = lvl_extrinsic.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
            lvl_extrinsic_inv = lvl_extrinsic_inv.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
            
            xs = torch.arange(pw).to(device) + 0.5
            ys = torch.arange(ph).to(device) + 0.5
            xs = xs * img_w / pw
            ys = ys * img_h / ph
            xs = xs[None, :].repeat(ph, 1)
            ys = ys[:, None].repeat(1, pw)
            img_coords = torch.stack([xs, ys], dim=-1)  # [ph, pw, 2]
            img_coords = img_coords.reshape(1, 1, ph*pw, 2)  # [1, 1, L_sf_ss_sv, 2]
            img_coords = img_coords.repeat(B, V, 1, 1)  # [B, V, L_sf_ss_sv, 2]
            img_coords = img_coords.reshape(B, -1, 2)  # [B, L_sf_ss, 2]
            frame_img_coords.append(img_coords)
            
            lvl_intrinsic = lvl_intrinsic.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
            lvl_extrinsic = lvl_extrinsic.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
            lvl_extrinsic_inv = lvl_extrinsic_inv.view(B, -1, 4, 4)  # [B, L_sf_ss, 4, 4]
                        
            frame_intrinsic.append(lvl_intrinsic)
            frame_extrinsic.append(lvl_extrinsic)
            frame_extrinsic_inv.append(lvl_extrinsic_inv)
            
            
        frame_intrinsic = torch.cat(frame_intrinsic, dim=1)  # B x L_sf x 4 x 4
        frame_extrinsic = torch.cat(frame_extrinsic, dim=1)  # B x L_sf x 4 x 4
        frame_extrinsic_inv = torch.cat(frame_extrinsic_inv, dim=1)  # B x L_sf x 4 x 4
        frame_img_coords = torch.cat(frame_img_coords, dim=1)  # B x L_sf x 2
        
        frame_intrinsic_norm = torch.zeros_like(frame_intrinsic)
        frame_intrinsic_norm[..., 0, 0] = frame_intrinsic[..., 0, 0] / img_w
        frame_intrinsic_norm[..., 1, 1] = frame_intrinsic[..., 1, 1] / img_h
        frame_intrinsic_norm[..., 0, 2] = frame_intrinsic[..., 0, 2] / img_w - 0.5
        frame_intrinsic_norm[..., 1, 2] = frame_intrinsic[..., 1, 2] / img_h - 0.5
        frame_intrinsic_norm[..., 2, 2] = 1
        frame_intrinsic_norm[..., 3, 3] = 1

        frame_meta_data = {
            "intrinsic": frame_intrinsic,
            "intrinsic_norm": frame_intrinsic_norm,
            "extrinsic": frame_extrinsic,
            "extrinsic_inv": frame_extrinsic_inv,
            "img_coords": frame_img_coords,
        }
        return frame_meta_data

    def _get_view_meta_data_multiframe(self, meta_data_list, scale_schedule):
        view_meta_data = {}
        first_to_curr_matrix = torch.eye(4).to(meta_data_list[0]['rot'].device)[None, None].repeat(meta_data_list[0]['rot'].shape[0], 1, 1, 1)
        curr_to_first_lidar = torch.eye(4).to(meta_data_list[0]['rot'].device)[None, None].repeat(meta_data_list[0]['rot'].shape[0], 1, 1, 1)
        # [B, 1, 4, 4]
        for tid, meta_data in enumerate(meta_data_list):
            # Generate rays for each frame
            cam_parameter = self._get_camera_parameters(meta_data)
            meta_data_frame = self._get_view_meta_data(cam_parameter, meta_data['size'], scale_schedule)
            # [B, L_sf, 4, 4]
            
            if 'curr_to_first_lidar' in meta_data:
                first_to_curr_matrix = torch.inverse(meta_data['curr_to_first_lidar'][:, None])
                curr_to_first_lidar = meta_data['curr_to_first_lidar'][:, None]
            elif tid != 0:
                first_to_curr_matrix = torch.inverse(meta_data['curr_to_prev_lidar'][:, None]) @ first_to_curr_matrix
                curr_to_first_lidar = curr_to_first_lidar @ meta_data['curr_to_prev_lidar'][:, None]
            meta_data_frame['extrinsic'] = meta_data_frame['extrinsic'] @ first_to_curr_matrix
            meta_data_frame['extrinsic_inv'] = curr_to_first_lidar @ meta_data_frame['extrinsic_inv']
            # meta_data_frame['extrinsic_norm'] = meta_data_frame['extrinsic'].clone()
            # meta_data_frame['extrinsic_norm'][:, :, :3, 3] = meta_data_frame['extrinsic_norm'][:, :, :3, 3] / 5
            meta_data_frame['timestep'] = meta_data['timestep'][:, None].repeat(1, meta_data_frame['extrinsic'].shape[1])
            
            img_coords = torch.cat([meta_data_frame['img_coords'], torch.ones_like(meta_data_frame['img_coords'][..., :2])], dim=-1)
            cam_coords =  torch.inverse(meta_data_frame['intrinsic'].to(torch.float32)) @ img_coords[..., None]
            world_coords = meta_data_frame['extrinsic_inv'] @ cam_coords
            world_coords = world_coords.squeeze(-1)
            camera_center_coords = meta_data_frame['extrinsic_inv'][..., :3, 3]
            ray = world_coords[..., :3] - camera_center_coords
            ray_norm = ray / torch.norm(ray, dim=-1, keepdim=True)
            
            meta_data_frame['ray'] = ray_norm
            meta_data_frame['camera_center'] = camera_center_coords
                        
            for key in meta_data_frame:
                if key not in view_meta_data:
                    view_meta_data[key] = []
                view_meta_data[key].append(meta_data_frame[key])

        for key in view_meta_data:
            view_meta_data[key] = torch.cat(view_meta_data[key], dim=1)
        return view_meta_data
    
    def _merge_bbox_map_condition(self, bbox_ca_kv, map_ca_kv):
        if bbox_ca_kv is None:
            return map_ca_kv
        if map_ca_kv is None:
            return bbox_ca_kv

        if self.use_condition_rope:
            bbox_kv_compact, bbox_cu_seqlens_k, bbox_center_compact = bbox_ca_kv
            map_kv_compact, map_cu_seqlens_k, map_center_compact = map_ca_kv
            assert bbox_center_compact.shape[0] == bbox_kv_compact.shape[0], f"{bbox_center_compact.shape[0]} != {bbox_kv_compact.shape[0]}"
            assert map_center_compact.shape[0] == map_kv_compact.shape[0], f"{map_center_compact.shape[0]} != {map_kv_compact.shape[0]}"
        else:
            bbox_kv_compact, bbox_cu_seqlens_k = bbox_ca_kv
            map_kv_compact, map_cu_seqlens_k = map_ca_kv
        
        merged_kv_compact = []
        if self.use_condition_rope:
            merged_center_compact = []
        merged_cu_seqlens_k = [0]
        
        assert len(bbox_cu_seqlens_k) == len(map_cu_seqlens_k)
        for i in range(len(bbox_cu_seqlens_k)-1):
            bbox_sid = bbox_cu_seqlens_k[i]
            bbox_eid = bbox_cu_seqlens_k[i+1]
            map_sid = map_cu_seqlens_k[i]
            map_eid = map_cu_seqlens_k[i+1]
            merged_kv_compact.append(bbox_kv_compact[bbox_sid:bbox_eid])
            merged_kv_compact.append(map_kv_compact[map_sid:map_eid])
            if self.use_condition_rope:
                merged_center_compact.append(bbox_center_compact[bbox_sid:bbox_eid])
                merged_center_compact.append(map_center_compact[map_sid:map_eid])
            merged_cu_seqlens_k.append(merged_cu_seqlens_k[-1] + bbox_eid - bbox_sid + map_eid - map_sid)
            
        merged_kv_compact = torch.cat(merged_kv_compact, dim=0)
        assert len(merged_cu_seqlens_k) == len(bbox_cu_seqlens_k)
        assert merged_cu_seqlens_k[-1] == bbox_cu_seqlens_k[-1] + map_cu_seqlens_k[-1]
        if self.use_condition_rope:
            merged_center_compact = torch.cat(merged_center_compact, dim=0)
            assert merged_center_compact.shape[0] == merged_kv_compact.shape[0], f"{merged_center_compact.shape[0]} != {merged_kv_compact.shape[0]}"
            return (merged_kv_compact, merged_cu_seqlens_k, merged_center_compact)
        else:
            return (merged_kv_compact, merged_cu_seqlens_k)
    
    def prepare_conditions(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, 
        scale_schedule: List[Tuple[int]], bbox_sequence=None, map_sequence=None, timesteps=None,**kwargs,
    ):

        if timesteps is None:
            timesteps = self.timesteps
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]
        
        L_single = int(np.sum([np.prod(s) for s in scale_schedule]))
        num_views = x_BLC_wo_prefix.shape[1] // ((L_single-1) * timesteps)
        assert x_BLC_wo_prefix.shape[1] % ((L_single-1) * timesteps) == 0
        
        # [1. get input sequence x_BLC]
        kv_compact_ori, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        # drop cond
        total = 0
        kv_compact = []
        for le in lens:
            if random.random() < self.cond_drop_rate:
                kv_compact.append(self.cfg_uncond[:le])
            else:
                kv_compact.append(kv_compact_ori[total:total+le])
            total += le
        kv_compact = torch.cat(kv_compact, dim=0)
        # must_on_graph = self.cfg_uncond[0, 0] * 0
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
        if sos.shape[0] == B * timesteps:
            sos = sos.reshape(B, timesteps, -1)  # [B, T, C]
            cond_BD = cond_BD.reshape(B, timesteps, -1)
            cond_BD = cond_BD[:, :, None].repeat(1, 1, L_single*num_views, 1)
            cond_BD = cond_BD.flatten(1, 2)  # [B, L, C]
        else:
            sos = sos[:, None].repeat(1, timesteps, 1)  # [B, T, C]
            cond_BD = cond_BD[:, None].repeat(1, timesteps*L_single*num_views, 1)  # [B, L, C]
            

        kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
        # kv_compact[0, 0] += must_on_graph
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

        bbox_ca_kv = None
        # Prepare bounding box conditions
        if self.object_condition:
            # Randomly drop bboxes
            if self.recurrent_training:
                sample_drop = np.zeros(B, dtype=bool)
            else:
                sample_drop = np.random.rand(B) < self.object_cond_drop_rate
            frame_drop =  np.random.rand(B, timesteps) < self.object_cond_drop_rate
            frame_drop = np.logical_or(sample_drop[:, None], frame_drop)
            
            # Replace dropped boxes with empty tokens
            all_bbox_features = torch.zeros((0, self.C)).type_as(kv_compact)
            if self.use_condition_rope:
                all_bbox_center = torch.zeros((0, 3)).type_as(kv_compact)
            all_seqlens_k = [0]
            all_seqlens_max = 0
            for tid in range(timesteps):
                for vid in range(num_views):
                    for bid in range(B):
                        if bbox_sequence[tid][2][vid][bid] == 0:
                            all_seqlens_k.append(all_seqlens_k[-1])
                        else:
                            # Encode bboxes
                            if not frame_drop[bid, tid]:
                                bbox_features = self.bbox_encoder(bbox_sequence[tid][0][vid][bid], bbox_sequence[tid][1][vid][bid].long())
                                all_bbox_features = torch.cat([all_bbox_features, bbox_features], dim=0)
                                if self.use_condition_rope:
                                    bbox_center = bbox_sequence[tid][3][vid][bid]
                                    all_bbox_center = torch.cat([all_bbox_center, bbox_center], dim=0)
                            else:
                                all_bbox_features = self.bbox_encoder.add_n_uncond_tokens(all_bbox_features, bbox_sequence[tid][2][vid][bid])
                                if self.use_condition_rope:
                                    pad_bbox_center = torch.Tensor([[0.5, 0.5, 1]]).to(bbox_sequence[tid][3][vid][bid].device)
                                    pad_bbox_center = pad_bbox_center.repeat(bbox_sequence[tid][2][vid][bid], 1)
                                    all_bbox_center = torch.cat([all_bbox_center, pad_bbox_center], dim=0)
                            all_seqlens_k.append(all_bbox_features.shape[0])
                            all_seqlens_max = max(all_seqlens_max, bbox_sequence[tid][2][vid][bid])

            if all_seqlens_k[-1] > 0:
                all_bbox_features = self.object_norm(all_bbox_features)
                all_bbox_features = self.object_proj_for_ca(all_bbox_features).contiguous()

            bbox_ca_kv = (all_bbox_features, all_seqlens_k)
            if self.use_condition_rope:
                bbox_ca_kv = (all_bbox_features, all_seqlens_k, all_bbox_center)
                    
        map_ca_kv = None
        if self.map_condition:
            # Randomly drop bboxes
            if self.recurrent_training:
                sample_drop = np.zeros(B, dtype=bool)
            else:
                sample_drop = np.random.rand(B) < self.map_condition_drop_rate
            frame_drop =  np.random.rand(B, timesteps) < self.map_condition_drop_rate

            if self.bbox_img_coord:
                all_map_points = torch.zeros((0, self.map_sample_points_num, 6)).type_as(kv_compact)
                all_map_points_mask = torch.zeros((0, self.map_sample_points_num)).type_as(kv_compact)
            else:
                all_map_points = torch.zeros((0, self.map_sample_points_num, 3)).type_as(kv_compact)
                all_map_points_mask = torch.zeros((0, self.map_sample_points_num)).type_as(kv_compact)
            if self.use_condition_rope:
                all_map_center = torch.zeros((0, 3)).type_as(kv_compact)
            all_map_labels = torch.zeros((0,)).type_as(cu_seqlens_k)
            all_map_seqlens_k = [0]
            all_map_seqlens_max = 0
            for tid in range(timesteps):
                for vid in range(num_views):
                    for bid in range(B):
                        if frame_drop[bid, tid]:
                            all_map_seqlens_k.append(all_map_seqlens_k[-1])
                            continue
                        map_elements_mask = np.random.rand(map_sequence[tid][2][vid][bid]) > self.map_condition_drop_rate
                        map_elements_num = np.sum(map_elements_mask)
                        if map_elements_num == 0:
                            all_map_seqlens_k.append(all_map_seqlens_k[-1])
                            continue
                        curr_map_points = map_sequence[tid][0][vid][bid][map_elements_mask]
                        curr_map_labels = map_sequence[tid][1][vid][bid][map_elements_mask]
                        curr_map_points_mask = map_sequence[tid][3][vid][bid][map_elements_mask]
                        all_map_points = torch.cat([all_map_points, curr_map_points], dim=0)
                        all_map_labels = torch.cat([all_map_labels, curr_map_labels], dim=0)
                        all_map_points_mask = torch.cat([all_map_points_mask, curr_map_points_mask], dim=0)
                        all_map_seqlens_k.append(all_map_labels.shape[0])
                        all_map_seqlens_max = max(all_map_seqlens_max, curr_map_labels.shape[0])
                        
                        if self.use_condition_rope:
                            map_center = map_sequence[tid][4][vid][bid][map_elements_mask]
                            all_map_center = torch.cat([all_map_center, map_center], dim=0)


            # Encode Map
            if all_map_points.shape[0] > 0:
                all_map_features = self.map_encoder(all_map_points, all_map_labels.long(), all_map_points_mask)
            else:
                all_map_features = torch.zeros((0, self.C)).type_as(kv_compact)
            
            if all_map_seqlens_k[-1] > 0:
                all_map_features = self.map_norm(all_map_features)
                all_map_features = self.map_proj_for_ca(all_map_features).contiguous()

            map_ca_kv = (all_map_features, all_map_seqlens_k)
            if self.use_condition_rope:
                map_ca_kv = (all_map_features, all_map_seqlens_k, all_map_center)                

        bbox_ca_kv = self._merge_bbox_map_condition(bbox_ca_kv, map_ca_kv)

        sos = sos.unsqueeze(2) + self.pos_1LsC_start[:, None].repeat(B, 1, num_views, 1)  # [B, T, V, C]

        return ca_kv, bbox_ca_kv, sos, cond_BD
    
    
    def prepare_spatial_temporal_embedding(self, scale_schedule, num_views, camera_meta_datas_list, need_to_pad, l_end, timesteps=None):
        if timesteps is None:
            timesteps = self.timesteps
        if 'ray' in self.view_embed_type:
            if 'plucker' in self.view_embed_type:
                plucker = True
            else:
                plucker = False
            if 'local' in self.view_embed_type:
                ray_BLD3 = self._generate_rays_multi(camera_meta_datas_list, scale_schedule, local=True, plucker=plucker)
            else:
                # Generate rays including ego-motion
                ray_BLD3 = self._generate_rays_multi(camera_meta_datas_list, scale_schedule, plucker=plucker)
            view_embedding_flatten = self.view_embed(ray_BLD3) # BT x sum(1*1*v + 2*2*v + ...) x D3
            view_embedding_flatten = torch.cat([view_embedding_flatten, torch.zeros_like(view_embedding_flatten[:, :need_to_pad])], dim=1)
        elif self.view_embed_type == 'none':
            view_embedding_flatten = None
        else:
            raise NotImplementedError

        # Generate time embedding
        if self.time_embed_type == 'fourier':
            time_embed = self._generate_fourier_time_embedding(camera_meta_datas_list, l_end // timesteps, need_to_pad, timesteps=timesteps)
        else:
            assert self.time_embed_type == 'none'
            time_embed = None
        
        return view_embedding_flatten, time_embed


    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False, camera_meta_datas_list=None, bbox_sequence=None, vae=None, cache=None, use_gt_idx=None, map_sequence=None, **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, **kwargs)
        if self.recurrent_training:
            return self.forward_recurrent(label_B_or_BLT, x_BLC_wo_prefix, scale_schedule, camera_meta_datas_list, bbox_sequence, cache, map_sequence, **kwargs)
        else:
            return self.forward_parallel(label_B_or_BLT, x_BLC_wo_prefix, scale_schedule, camera_meta_datas_list, bbox_sequence, map_sequence, **kwargs)

    def forward_recurrent(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        camera_meta_datas_list=None, bbox_sequence=None, cache=None, map_sequence=None, **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """

        timesteps = 1
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]
        
        L_single = int(np.sum([np.prod(s) for s in scale_schedule]))
        num_views = x_BLC_wo_prefix.shape[1] // (L_single - 1)
        assert x_BLC_wo_prefix.shape[1] % (L_single-1) == 0

        
        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            ca_kv, bbox_ca_kv, sos, cond_BD = self.prepare_conditions(label_B_or_BLT, x_BLC_wo_prefix, scale_schedule, bbox_sequence, map_sequence, timesteps=1)
            if bbox_ca_kv is not None:
                ca_kv = self._merge_text_bbox_condition(ca_kv, bbox_ca_kv, B)
            elif (self.object_condition or self.map_condition) and self.use_condition_rope:
                text_kv_compact, text_cu_seqlens_k, text_max_seqlen_k = ca_kv
                text_center_compact = torch.Tensor([[0.5, 0.5, 1]]).type_as(text_kv_compact).repeat(text_kv_compact.shape[0], 1)
                ca_kv = (text_kv_compact, text_cu_seqlens_k, text_max_seqlen_k, text_center_compact)
                

            x_BLC = []
            embeded_x_BLC = self.word_embed(self.norm0_ve(x_BLC_wo_prefix)) # B x L x C
            embeded_x_BLC = embeded_x_BLC.view(B, timesteps, -1, self.C) # B x T x L x C
            for tid in range(timesteps):
                x_BLC.append(sos[:, tid])
                x_BLC.append(embeded_x_BLC[:, tid])
            x_BLC = torch.cat(x_BLC, dim=1)

            if cache is None:
                cache = {}
                for block_idx in range(len(self.block_chunks)):
                    cache[block_idx] = None

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0

            # AdaLN inputs
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            if self.use_temporal_attn:
                cond_BD_or_gss_temporal = self.shared_ada_lin_temporal(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            else:
                cond_BD_or_gss_temporal = None

            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                # Prepare the attention masks for scale, spatial, and temporal blocks
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2]*num_views, ), i) for i, pn in enumerate(scale_schedule)] * timesteps).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                if self.attn_bias_type == 'separate':
                    raise NotImplementedError
                elif self.attn_bias_type == 'full':
                    raise NotImplementedError
                elif self.attn_bias_type == 'temporal_scale':
                    attn_bias_for_masking = self._attn_to_all_previous_low_scale(attn_bias_for_masking, l_end // timesteps)
                elif self.attn_bias_type == 'temporal_same_scale':
                    attn_bias_for_masking_same_scale = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_for_masking = self._attn_to_all_previous_low_scale(attn_bias_for_masking_same_scale, l_end // timesteps)
                elif self.attn_bias_type == 'self':
                    attn_bias_for_masking = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_for_masking = self._attn_to_self(attn_bias_for_masking, l_end // timesteps)
                    # # debugging
                    # import numpy as np
                    # import matplotlib.pyplot as plt
                    # print("attn_bias_for_masking", attn_bias_for_masking.shape)
                    # mask_np = attn_bias_for_masking[0, 0].cpu().numpy()
                    # mask_np = np.where(mask_np == -np.inf, -1, mask_np)  # Replace -inf for visualization
                    # plt.imshow(mask_np, cmap="gray", interpolation='nearest')   
                    # # plt.show()
                    # plt.savefig("debug/mask.png") 
                    # exit(0)
                elif self.attn_bias_type == 'markov':
                    attn_bias_same_scale = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_last_scale = torch.where(d == len(scale_schedule)-1, 0., -torch.inf).reshape(1, 1, 1, l_end).repeat(1, 1, l_end, 1)
                    attn_bias_for_masking = self._attn_mask_markov(attn_bias_same_scale, attn_bias_last_scale, l_end // timesteps)
                else:
                    raise NotImplementedError


                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2], ), i) for i, pn in enumerate(scale_schedule)]).view(1, L_single, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                if self.attn_bias_type == 'markov':
                    attn_bias_for_masking_sa = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, L_single, L_single)
                else:
                    attn_bias_for_masking_sa = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L_single, L_single)

                
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                attn_bias_sa = attn_bias_for_masking_sa[:, :, :L_single, :L_single].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    
                    attn_bias_sa = F.pad(attn_bias_sa, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias_sa[0, 0, L_single:, 0] = 0
                    
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector_sa = attn_bias_sa.type_as(x_BLC).to(x_BLC.device)
                attn_bias_or_two_vector_temporal = attn_bias.type_as(x_BLC).to(x_BLC.device)
                attn_bias_or_two_vector_dict = {
                    'sa': attn_bias_or_two_vector_sa,
                    'temporal': attn_bias_or_two_vector_temporal,
                }

        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        view_embedding_flatten, time_embed = self.prepare_spatial_temporal_embedding(scale_schedule, num_views, camera_meta_datas_list[-1:], need_to_pad, l_end)
        
        if 'prope' in self.view_embed_type:
            view_meta_data = self._get_view_meta_data_multiframe(camera_meta_datas_list, scale_schedule)
            if 'camT' in self.view_embed_type:
                view_meta_data['remove_xy'] = True
            else:
                view_meta_data['remove_xy'] = False
            view_meta_data['view_embed_type'] = self.view_embed_type
        else:
            view_meta_data = None

        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
                
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if view_embedding_flatten is not None:
                    if self.add_view_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + view_embedding_flatten
                    if not self.add_view_embeding_only_first_block:
                        x_BLC = x_BLC + view_embedding_flatten
                if time_embed is not None:
                    if self.add_time_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + time_embed
                    if not self.add_time_embeding_only_first_block:
                        x_BLC = x_BLC + time_embed
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector_dict, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector_dict, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal, past_cache=cache[i])
                if isinstance(x_BLC, tuple):
                    x_BLC, cache_chunk = x_BLC
                    del cache[i]
                    cache[i] = cache_chunk
        
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if view_embedding_flatten is not None:
                    if self.add_view_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + view_embedding_flatten
                    if not self.add_view_embeding_only_first_block:
                        x_BLC = x_BLC + view_embedding_flatten
                if time_embed is not None:
                    if self.add_time_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + time_embed
                    if not self.add_time_embeding_only_first_block:
                        x_BLC = x_BLC + time_embed
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views)             
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector_dict, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid, num_views=num_views, timesteps=self.timesteps, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal, past_cache=cache[i])

                if isinstance(x_BLC, tuple):
                    x_BLC, cache_chunk = x_BLC
                    del cache[i]
                    cache[i] = cache_chunk

        # [3. unpad the seqlen dim, and then get logits]
        logits = self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size
        return logits, cache

    def forward_parallel(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        camera_meta_datas_list=None, bbox_sequence=None, map_sequence=None, **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV
        """
        label_B_or_BLT: label_B or (kv_compact, cu_seqlens_k, max_seqlen_k)
        :return: logits BLV, V is vocab_size
        """
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]
        
        L_single = int(np.sum([np.prod(s) for s in scale_schedule]))
        num_frames = len(camera_meta_datas_list)
        num_views = x_BLC_wo_prefix.shape[1] // ((L_single-1) * num_frames)

        assert x_BLC_wo_prefix.shape[1] % ((L_single-1) * num_frames) == 0
                
        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            ca_kv, bbox_ca_kv, sos, cond_BD = self.prepare_conditions(label_B_or_BLT, x_BLC_wo_prefix, scale_schedule, bbox_sequence, map_sequence, timesteps=num_frames)
            if bbox_ca_kv is not None:
                ca_kv = self._merge_text_bbox_condition(ca_kv, bbox_ca_kv, B, timesteps=num_frames, num_views=num_views)
            elif (self.object_condition or self.map_condition) and self.use_condition_rope:
                text_kv_compact, text_cu_seqlens_k, text_max_seqlen_k = ca_kv
                text_center_compact = torch.Tensor([[0.5, 0.5, 1]]).type_as(text_kv_compact).repeat(text_kv_compact.shape[0], 1)
                ca_kv = (text_kv_compact, text_cu_seqlens_k, text_max_seqlen_k, text_center_compact)
                

            x_BLC = []
            embeded_x_BLC = self.word_embed(self.norm0_ve(x_BLC_wo_prefix)) # B x L x C
            embeded_x_BLC = embeded_x_BLC.view(B, num_frames, -1, self.C) # B x T x L x C
            for tid in range(num_frames):
                x_BLC.append(sos[:, tid])
                x_BLC.append(embeded_x_BLC[:, tid])
            x_BLC = torch.cat(x_BLC, dim=1)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0

            # AdaLN inputs
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            if self.use_temporal_attn:
                cond_BD_or_gss_temporal = self.shared_ada_lin_temporal(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            else:
                cond_BD_or_gss_temporal = None

            if self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
                attn_bias_or_two_vector_dict = {"sa": None, "temporal": None}
            else:
                # Prepare the attention masks for scale, spatial, and temporal blocks
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2]*num_views, ), i) for i, pn in enumerate(scale_schedule)] * num_frames).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                if self.attn_bias_type == 'separate':
                    attn_bias_for_masking = self._get_separate_attn_bias(attn_bias_for_masking, scale_schedule)
                elif self.attn_bias_type == 'independent':
                    attn_bias_for_masking = self._get_independent_attn_bias(attn_bias_for_masking, scale_schedule, num_views, l_end // num_frames, num_frames)
                elif self.attn_bias_type == 'full':
                    attn_bias_for_masking = self._attn_to_all_previous(attn_bias_for_masking, l_end // num_frames, num_frames)
                elif self.attn_bias_type == 'temporal_scale':
                    attn_bias_for_masking = self._attn_to_all_previous_low_scale(attn_bias_for_masking, l_end // num_frames, num_frames)
                elif self.attn_bias_type == 'temporal_same_scale':
                    attn_bias_for_masking_same_scale = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_for_masking = self._attn_to_all_previous_low_scale(attn_bias_for_masking_same_scale, l_end // num_frames, num_frames)
                elif self.attn_bias_type == 'self':
                    attn_bias_for_masking = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_for_masking = self._attn_to_self(attn_bias_for_masking, l_end // num_frames, num_frames)
                elif self.attn_bias_type == 'markov':
                    attn_bias_same_scale = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                    attn_bias_last_scale = torch.where(dT == len(scale_schedule)-1, 0., -torch.inf).reshape(1, 1, 1, l_end).repeat(1, 1, l_end, 1)
                    attn_bias_for_masking = self._attn_mask_markov(attn_bias_same_scale, attn_bias_last_scale, l_end // num_frames, num_frames)
                else:
                    raise NotImplementedError


                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2], ), i) for i, pn in enumerate(scale_schedule)]).view(1, L_single, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                if self.attn_bias_type == 'markov':
                    attn_bias_for_masking_sa = torch.where(d == dT, 0., -torch.inf).reshape(1, 1, L_single, L_single)
                else:
                    attn_bias_for_masking_sa = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L_single, L_single)

                # # debugging
                # mask_np = attn_bias_for_masking[0, 0].cpu().numpy()
                # mask_np = np.where(mask_np == -np.inf, -1, mask_np)  # Replace -inf for visualization
                # plt.imshow(mask_np, cmap="gray", interpolation='nearest')   
                # plt.savefig("mask.png") 
                # mask_np = attn_bias_for_masking_sa[0, 0].cpu().numpy()
                # mask_np = np.where(mask_np == -np.inf, -1, mask_np)  # Replace -inf for visualization
                # plt.imshow(mask_np, cmap="gray", interpolation='nearest')   
                # plt.savefig("mask_sa.png") 
                
                # exit(0)
                
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                attn_bias_sa = attn_bias_for_masking_sa[:, :, :L_single, :L_single].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    
                    attn_bias_sa = F.pad(attn_bias_sa, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias_sa[0, 0, L_single:, 0] = 0
                    
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector_sa = attn_bias_sa.type_as(x_BLC).to(x_BLC.device)
                attn_bias_or_two_vector_temporal = attn_bias.type_as(x_BLC).to(x_BLC.device)
                attn_bias_or_two_vector_dict = {
                    'sa': attn_bias_or_two_vector_sa,
                    'temporal': attn_bias_or_two_vector_temporal,
                }

        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = ({nv: None for nv in range(1, self.num_views+1)}, {nv: None for nv in range(1, self.num_views+1)})

        view_embedding_flatten, time_embed = self.prepare_spatial_temporal_embedding(scale_schedule, num_views, camera_meta_datas_list, need_to_pad, l_end, num_frames)
                
        if 'prope' in self.view_embed_type:
            view_meta_data = self._get_view_meta_data_multiframe(camera_meta_datas_list, scale_schedule)
            if 'camT' in self.view_embed_type:
                view_meta_data['remove_xy'] = True
            else:
                view_meta_data['remove_xy'] = False
            view_meta_data['view_embed_type'] = self.view_embed_type
        else:
            view_meta_data = None

        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
                
        if self.num_block_chunks == 1:
            for i, b in enumerate(self.blocks):
                if view_embedding_flatten is not None:
                    if self.add_view_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + view_embedding_flatten
                    if not self.add_view_embeding_only_first_block:
                        x_BLC = x_BLC + view_embedding_flatten
                if time_embed is not None:
                    if self.add_time_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + time_embed
                    if not self.add_time_embeding_only_first_block:
                        x_BLC = x_BLC + time_embed
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views, timesteps=num_frames)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views, timesteps=num_frames)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector_dict, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector_dict, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal, timesteps=num_frames)
        else:
            for i, chunk in enumerate(self.block_chunks): # this path
                if view_embedding_flatten is not None:
                    if self.add_view_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + view_embedding_flatten
                    if not self.add_view_embeding_only_first_block:
                        x_BLC = x_BLC + view_embedding_flatten
                if time_embed is not None:
                    if self.add_time_embeding_only_first_block and i == 0:
                        x_BLC = x_BLC + time_embed
                    if not self.add_time_embeding_only_first_block:
                        x_BLC = x_BLC + time_embed
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views, timesteps=num_frames)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad, num_views=num_views, timesteps=num_frames)             
                x_BLC = chunk(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector_dict, attn_fn=attn_fn, scale_schedule=scale_schedule, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=self.rope2d_freqs_grid, num_views=num_views, timesteps=num_frames, view_meta_data=view_meta_data, cond_BD_temporal=cond_BD_or_gss_temporal)

        # [3. unpad the seqlen dim, and then get logits]
        logits = self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size
        
        # print("num_frames: ", num_frames, "num_views: ", num_views, "l_end: ", l_end)
        return logits
    
    def _generate_ray_embedding_frame(self, ray_BLD3, curr_to_prev, local=False, plucker=False):
        if local:
            ray_pos = ray_BLD3.clone()
        else:
            ray_BLD3_pad = torch.cat([ray_BLD3, torch.ones_like(ray_BLD3[...,:1])], dim=-1)  # [B, L, D, 4]
            ray_BLD3_pad = torch.matmul(curr_to_prev[:, None, None], ray_BLD3_pad[..., None])  # [B, L, D, 4, 1]
            ray_pos = ray_BLD3_pad.squeeze(-1)[..., :3].clone()  # [B, L, D, 3]

        if plucker:
            cam_pos = ray_pos[:, :, 0]  # [B, L, 3]
            ray_dir = ray_pos[:, :, 10] - cam_pos  # [B, L, 3]
            ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)
            ray_m = torch.cross(ray_dir, cam_pos, dim=-1)  # [B, L, 3]
            ray_pos = torch.cat([ray_m, ray_dir], dim=-1)  # [B, L, 6]
        else:
            for i in range(3):
                ray_pos[..., i] = ray_pos[..., i] / self.ray_normalize[i]
            ray_pos = ray_pos.flatten(2,3).clone()  # [B, L, D*3]
        ray_embedding = self.view_embed(ray_pos) # BT x sum(1*1*v + 2*2*v + ...) x D3
        return ray_embedding


    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        temporal_gt_leak=0, temporal_gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        camera_meta_datas_list=None,
        bbox_sequence=None,
        map_sequence=None,
        exit_step=10,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2]*self.num_views, ), i) for i, pn in enumerate(scale_schedule)] * self.timesteps).view(1, -1, 1)
        l_end = d.shape[1]
        l_single = l_end // self.timesteps
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            
            for camera_meta_data in camera_meta_datas_list:
                for key in ['rot', 'trans', 'intrins', 'post_rot', 'post_trans', 'curr_to_prev_lidar', 'timestep']:
                    camera_meta_data[key] = torch.cat([camera_meta_data[key], camera_meta_data[key]], dim=0)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        sos = sos.reshape(bs, self.timesteps, -1)  # [B, T, C]
        cond_BD = cond_BD[:, None].repeat(1, l_single, 1).reshape(bs, l_single*self.timesteps, -1)  # [B, L, C]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        text_ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k

        need_to_pad = 0
        
        start_token = sos.unsqueeze(2) + self.pos_1LsC_start[:, None].repeat(bs, 1, self.num_views, 1)  # [B, T, V, C]
        with torch.amp.autocast('cuda', enabled=False):
            all_cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
            if self.use_temporal_attn:
                all_cond_BD_or_gss_temporal = self.shared_ada_lin_temporal(cond_BD.float()).float().contiguous()
            else:
                all_cond_BD_or_gss_temporal = None
        all_cond_BD = cond_BD.clone()
        del cond_BD
        
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []

        caching_use_dict = self.attn_bias_type != 'full' and self.attn_bias_type != 'markov'
        prefix_cache = self.attn_bias_type == 'temporal_scale'
        last_only = self.attn_bias_type == 'markov'
        if inference_mode:
            for b in self.unregistered_blocks: 
                if self.attn_bias_type != 'markov':
                    b.sa.kv_caching(True)
                if self.use_temporal_attn:
                    b.sa_temporal.kv_caching(True, caching_use_dict=caching_use_dict, prefix_cache=prefix_cache, last_scale_only=last_only)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    if self.attn_bias_type != 'markov':
                        module.sa.kv_caching(True)
                    if self.use_temporal_attn:
                        module.sa_temporal.kv_caching(True, caching_use_dict=caching_use_dict, prefix_cache=prefix_cache, last_scale_only=last_only)
                    
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        if 'ray' in self.view_embed_type:
            ray_BLD3 = self._generate_rays(camera_meta_datas_list[0], scale_schedule, normalize=False)
            L_per_step = ray_BLD3.shape[1]
            ray_BLD3 = ray_BLD3.reshape(bs, L_per_step, self.max_ray_depth, 3)
        elif self.view_embed_type == 'none':
            L_per_step = np.sum([np.prod(item) for item in scale_schedule]) * self.num_views
        else:
            raise NotImplementedError
        
        if self.time_embed_type == 'fourier':
            time_embed = self._generate_fourier_time_embedding(camera_meta_datas_list, L_per_step, need_to_pad)
        else:
            assert self.time_embed_type == 'none'
            time_embed = torch.zeros(1, L_per_step*self.timesteps, 1).type_as(kv_compact)


        if 'prope' in self.view_embed_type:
            view_meta_data_all = self._get_view_meta_data_multiframe(camera_meta_datas_list, scale_schedule)
        else:
            view_meta_data_all = None

        outputs = []
        curr_to_first_lidar = torch.eye(4)[None].repeat(bs, 1, 1).type_as(start_token)
        device = start_token.device
        for t in range(self.timesteps):
            last_stage = start_token[:, t]
            if inference_mode:
                for b in self.unregistered_blocks: 
                    if self.attn_bias_type != 'markov':
                        b.sa.kv_caching(True)
                    if self.use_temporal_attn:
                        b.sa_temporal.clean_cfg_cache()
            else:
                assert self.num_block_chunks > 1
                for block_chunk_ in self.block_chunks:
                    for module in block_chunk_.module.module:
                        if self.attn_bias_type != 'markov':
                            module.sa.kv_caching(True)
            
            cur_L_view = 0
            if 'ray' in self.view_embed_type:
                if 'plucker' in self.view_embed_type:
                    plucker = True
                else:
                    plucker = False
                if 'local' in self.view_embed_type:
                    current_view_embedding = self._generate_ray_embedding_frame(ray_BLD3, curr_to_first_lidar, local=True, plucker=plucker)  # [B, L, C]
                else:
                    current_view_embedding = self._generate_ray_embedding_frame(ray_BLD3, curr_to_first_lidar, plucker=plucker)  # [B, L, C]
            else:
                assert self.view_embed_type == 'none'
                current_view_embedding = torch.zeros([1, L_per_step, 1]).type_as(kv_compact)

            bbox_ca_kv = None
            if self.object_condition:
                box_mask_batch = []
                box_coord_img_batch = []
                if self.use_condition_rope:
                    box_center_img_batch = []
                frame_seqlens_k = [0]
                frame_seqlens_max = 0
                for bid in range(B):
                    if len(bbox_sequence[t][bid][0]) > 0:
                        frame_meta_data = {}
                        for key in camera_meta_datas_list[t]:
                            if key not in ['size', 'curr_to_prev_lidar']:
                                frame_meta_data[key] = camera_meta_datas_list[t][key][bid]
                        frame_meta_data['size'] = camera_meta_datas_list[t]['size']
                        box_mask, box_coord_img = project_corners_to_views(bbox_sequence[t][bid][0], torch.inverse(curr_to_first_lidar[bid]), frame_meta_data, return_coord_2d=True)
                        box_mask_batch.append(box_mask)
                        box_coord_img_batch.append(box_coord_img)
                        if self.use_condition_rope:
                            box_center = torch.mean(bbox_sequence[t][bid][0], dim=-2, keepdim=True)
                            _, box_center_img = project_corners_to_views(box_center, torch.inverse(curr_to_first_lidar[bid]), frame_meta_data, return_coord_2d=True)
                            box_center_img_batch.append(box_center_img.squeeze(1))
                    else:
                        box_mask_batch.append(torch.zeros((0, self.num_views)).to(device))
                        
                frame_bbox_features = torch.zeros((0, self.C)).to(device)
                if self.use_condition_rope:
                    frame_bbox_centers = torch.zeros((0, 3)).to(device)
                for vid in range(self.num_views):
                    for bid in range(B):
                        box_mask = box_mask_batch[bid]
                        view_mask = box_mask[:, vid]
                        if view_mask.sum() > 0:
                            view_corners = bbox_sequence[t][bid][0][view_mask]
                            view_labels = bbox_sequence[t][bid][1][view_mask]

                            if self.use_frame_coordinate:
                                view_corners = convert_points(view_corners, torch.inverse(curr_to_first_lidar[bid])[:3,:3], torch.inverse(curr_to_first_lidar[bid])[:3,3])

                            if self.bbox_img_coord:
                                view_corners_img = box_coord_img_batch[bid][view_mask, :, vid]
                                view_corners = torch.cat([view_corners, view_corners_img], dim=-1)

                            view_bbox_features = self.bbox_encoder(view_corners, view_labels)
                            frame_bbox_features = torch.cat([frame_bbox_features, view_bbox_features], dim=0)
                            frame_seqlens_k.append(frame_bbox_features.shape[0])
                            frame_seqlens_max = max(frame_seqlens_max, view_labels.shape[0])
                            if self.use_condition_rope:
                                view_center_img = box_center_img_batch[bid][view_mask, vid]
                                frame_bbox_centers = torch.cat([frame_bbox_centers, view_center_img], dim=0)
                        else:
                            frame_seqlens_k.append(frame_seqlens_k[-1])
                    for bid in range(B):
                        box_mask = box_mask_batch[bid]
                        view_mask = box_mask[:, vid]
                        if any(np.array(cfg_list) != 1):
                            if view_mask.sum() > 0:
                                frame_bbox_features = self.bbox_encoder.add_n_uncond_tokens(frame_bbox_features, view_mask.sum())
                                if self.use_condition_rope:
                                    pad_bbox_center = torch.Tensor([[0.5, 0.5, 1]]).to(device)
                                    pad_bbox_center = pad_bbox_center.repeat(view_mask.sum(), 1)
                                    frame_bbox_centers = torch.cat([frame_bbox_centers, pad_bbox_center], dim=0)
                            frame_seqlens_k.append(frame_bbox_features.shape[0])

                frame_bbox_features = self.object_norm(frame_bbox_features)
                frame_bbox_features = self.object_proj_for_ca(frame_bbox_features).contiguous()

                bbox_ca_kv = [frame_bbox_features, frame_seqlens_k]
                if self.use_condition_rope:
                    bbox_ca_kv.append(frame_bbox_centers)

            map_ca_kv = None
            if self.map_condition:
                map_points_mask_batch = []
                map_points_coord_img_batch = []
                if self.use_condition_rope:
                    map_center_img_batch = []
                frame_map_seqlens_k = [0]
                frame_map_seqlens_max = 0
                for bid in range(B):
                    if len(map_sequence[t][bid][0]) > 0:
                        frame_meta_data = {}
                        for key in camera_meta_datas_list[t]:
                            if key not in ['size', 'curr_to_prev_lidar']:
                                frame_meta_data[key] = camera_meta_datas_list[t][key][bid]
                        frame_meta_data['size'] = camera_meta_datas_list[t]['size']
                        point_mask, point_coord_img = project_corners_to_views(map_sequence[t][bid][0], torch.inverse(curr_to_first_lidar[bid]), frame_meta_data, return_coord_2d=True, pointwise_mask=True)
                        # point_mask[N, n_points, V]
                        # point_coord_img[N, n_points, V, 3]
                        map_points_mask_batch.append(point_mask)
                        map_points_coord_img_batch.append(point_coord_img)
                        if self.use_condition_rope:
                            center_point = torch.sum(map_sequence[t][bid][0][:, :, None] * point_mask[..., None], dim=1) / (torch.sum(point_mask[..., None], dim=1) + 1e-6)
                            # center_point: [N, V, 3]
                            _, point_center_img = project_corners_to_views(center_point, torch.inverse(curr_to_first_lidar[bid]), frame_meta_data, return_coord_2d=True, pointwise_mask=True)
                            # point_center_img: [N, V, V, 3]
                            map_center_img_batch.append(point_center_img)
                    else:
                        map_points_mask_batch.append(torch.zeros((0, self.map_sample_points_num, self.num_views)).to(device))
                    
                if self.bbox_img_coord:
                    frame_map_points = torch.zeros((0, self.map_sample_points_num, 6)).to(device)

                else:
                    frame_map_points = torch.zeros((0, self.map_sample_points_num, 3)).to(device)
                if self.use_condition_rope:
                    frame_map_centers = torch.zeros((0, 3)).to(device)
                frame_map_points_masks = torch.zeros((0, self.map_sample_points_num)).to(device)
                frame_map_labels = torch.zeros(0, dtype=torch.long).to(device)
                for vid in range(self.num_views):
                    for bid in range(B):
                        map_point_mask = map_points_mask_batch[bid]
                        view_map_point_mask = map_point_mask[:, :, vid]
                        view_mask = torch.max(view_map_point_mask, dim=1)[0]
                        if view_mask.sum() > 0:
                            view_map_points = map_sequence[t][bid][0][view_mask]
                            view_map_labels = map_sequence[t][bid][1][view_mask]
                            if self.use_frame_coordinate:
                                view_map_points = convert_points(view_map_points, torch.inverse(curr_to_first_lidar[bid])[:3,:3], torch.inverse(curr_to_first_lidar[bid])[:3,3])
                            if self.bbox_img_coord:
                                view_map_points_img = map_points_coord_img_batch[bid][view_mask, :, vid]
                                view_map_points = torch.cat([view_map_points, view_map_points_img], dim=-1)
                            
                            frame_map_points = torch.cat([frame_map_points, view_map_points], dim=0)
                            frame_map_labels = torch.cat([frame_map_labels, view_map_labels], dim=0)
                            frame_map_seqlens_k.append(frame_map_labels.shape[0])
                            frame_map_seqlens_max = max(frame_map_seqlens_max, view_map_labels.shape[0])
                            if self.use_condition_rope:
                                view_map_centers = map_center_img_batch[bid][view_mask, vid, vid]
                                frame_map_centers = torch.cat([frame_map_centers, view_map_centers], dim=0)
                            frame_map_points_masks = torch.cat([frame_map_points_masks, view_map_point_mask[view_mask]], dim=0)
                        else:
                            frame_map_seqlens_k.append(frame_map_seqlens_k[-1])
                    for bid in range(B):
                        if any(np.array(cfg_list) != 1):   
                            frame_map_seqlens_k.append(frame_map_seqlens_k[-1])

                if frame_map_labels.shape[0] > 0:
                    frame_map_features =  self.map_encoder(frame_map_points, frame_map_labels, frame_map_points_masks)
                else:
                    frame_map_features = torch.zeros((0, self.C)).to(device)
                
                frame_map_features = self.map_norm(frame_map_features)
                frame_map_features = self.map_proj_for_ca(frame_map_features).contiguous()

                map_ca_kv = [frame_map_features, frame_map_seqlens_k]
                if self.use_condition_rope:
                    map_ca_kv.append(frame_map_centers)
            
            if self.object_condition or self.map_condition:
                bbox_ca_kv = self._merge_bbox_map_condition(bbox_ca_kv, map_ca_kv)
            else:
                bbox_ca_kv = None
                
            text_kv_compact, text_cu_seqlens_k, text_max_seqlen_k = text_ca_kv
            text_kv_compact_frame = []
            text_lens_frame = [0]
            for bid in range(bs):
                start_text_id = text_cu_seqlens_k[bid*self.timesteps+t]
                end_text_id = text_cu_seqlens_k[bid*self.timesteps+t+1]
                text_kv_compact_frame.append(text_kv_compact[start_text_id:end_text_id])
                text_lens_frame.append(end_text_id - start_text_id)
            text_kv_compact_frame = torch.cat(text_kv_compact_frame, dim=0)
            text_lens_frame = torch.tensor(text_lens_frame, device=device)
            text_cu_seqlens_k_frame = torch.cumsum(text_lens_frame, dim=0).to(torch.int32)
            text_max_seqlen_k_frame = torch.max(text_lens_frame)

            text_ca_kv_frame = (text_kv_compact_frame, text_cu_seqlens_k_frame, text_max_seqlen_k_frame)
            ca_kv = self._merge_text_bbox_condition(text_ca_kv_frame, bbox_ca_kv, bs, self.num_views)            

            frame_cond_BD_or_gss = all_cond_BD_or_gss[:, t*l_single:(t+1)*l_single].contiguous()
            frame_cond_BD = all_cond_BD[:, t*l_single:(t+1)*l_single].contiguous()
            if self.use_temporal_attn:
                frame_cond_BD_or_gss_temporal = all_cond_BD_or_gss_temporal[:, t*l_single:(t+1)*l_single].contiguous()

            summed_codes = 0
            # print(f"timestep {t}")
            # print("last_stage", last_stage.shape)
            # print("view_embedding_flatten", view_embedding_flatten[:, cur_L:cur_L+1, :10])
            # print("time_embed", time_embed[:, cur_L:cur_L+1, :10])
            for si, pn in enumerate(scale_schedule):   # si: i-th segment
                last_scale = si == len(scale_schedule) - 1
                
                cfg = cfg_list[si]
                if si >= trunk_scale:
                    break
                start_L = cur_L
                cur_L += np.array(pn).prod() * self.num_views
                start_L_view = cur_L_view
                cur_L_view += np.array(pn).prod() * self.num_views

                view_meta_data = {}
                if view_meta_data_all is not None:
                    for key in view_meta_data_all:
                        view_meta_data[key] = view_meta_data_all[key][:, start_L:cur_L]
                    if 'camT' in self.view_embed_type:
                        view_meta_data['remove_xy'] = True
                    else:
                        view_meta_data['remove_xy'] = False
                    view_meta_data['view_embed_type'] = self.view_embed_type
                else:
                    view_meta_data = None


                cond_BD_or_gss = frame_cond_BD_or_gss[:, start_L_view:cur_L_view]  # [B, L_scale, C]
                cond_BD = frame_cond_BD[:, start_L_view:cur_L_view]  # [B, L_scale, C]
                if self.use_temporal_attn:
                    cond_BD_or_gss_temporal = frame_cond_BD_or_gss_temporal[:, start_L_view:cur_L_view]  # [B, L_scale, C]
                else:
                    cond_BD_or_gss_temporal = None

                # print(start_L, cur_L)
                attn_fn = None
                if self.use_flex_attn:
                    # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                    # if need_to_pad:
                    #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                    attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                # #TODO: implement non-full attention

                layer_idx = 0
                for block_idx, b in enumerate(self.block_chunks):
                    #NOTE: add embedding everytime?
                    if self.add_view_embeding_only_first_block and block_idx == 0:
                        last_stage = last_stage + current_view_embedding[:, start_L_view:cur_L_view]
                    if not self.add_view_embeding_only_first_block:
                        last_stage = last_stage + current_view_embedding[:, start_L_view:cur_L_view]
                    if self.add_time_embeding_only_first_block and block_idx == 0:
                        last_stage = last_stage + time_embed[:, start_L:cur_L]
                    if not self.add_time_embeding_only_first_block:
                        last_stage = last_stage + time_embed[:, start_L:cur_L]
                    # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block: 
                        last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)

                    for m in b.module:
                        last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si, timesteps=1, num_views=self.num_views, cond_BD_temporal=cond_BD_or_gss_temporal, view_meta_data=view_meta_data)
                        
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                            last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1
                
                if (cfg != 1) and add_cfg_on_logits:
                    # print(f'add cfg on add_cfg_on_logits')
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]

                else:
                    logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
                                  
                logits_BlV = logits_BlV.reshape(B*self.num_views, logits_BlV.shape[1]//self.num_views, logits_BlV.shape[2])
                if self.use_bit_label:
                    tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                    logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                    idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                    idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
                else:
                    idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]

                if vae_type != 0:
                    assert returns_vemb
                    # if si < gt_leak:
                    #     idx_Bld = gt_ls_Bl[si]
                    if t < temporal_gt_leak and (si < gt_leak or gt_leak==0):
                        idx_Bld = temporal_gt_ls_Bl[t][si].clone()
                        
                        # idx_Bld = temporal_gt_ls_Bl[t][si].clone()
                    else:
                        assert pn[0] == 1
                        idx_Bld = idx_Bld.reshape(B*self.num_views,pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                        if self.apply_spatial_patchify: # unpatchify operation
                            idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                            idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                            idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                    idx_Bld_list.append(idx_Bld)

                    codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    if not last_scale:
                        summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                        last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                        last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                        if self.apply_spatial_patchify: # patchify operation
                            last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                        last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                    else:
                        summed_codes += codes
                else:
                    # if si < gt_leak:
                    #     idx_Bl = gt_ls_Bl[si]
                    if t < temporal_gt_leak and (si < gt_leak or gt_leak==0):
                        idx_Bl = temporal_gt_ls_Bl[t][si]

                    h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                    # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                    h_BChw = h_BChw.transpose_(1, 2).reshape(B*self.num_views, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                    ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                    idx_Bl_list.append(idx_Bl)
                    if si != num_stages_minus_1:
                        accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
                
                if si != num_stages_minus_1:
                    last_stage = self.word_embed(self.norm0_ve(last_stage))
                    last_stage = last_stage.repeat(bs//B, 1, 1)
                    last_stage = last_stage.reshape(bs, self.num_views*last_stage.shape[1], last_stage.shape[2])
                else: # prepare for next timestep
                    last_stage = start_token # - time_embed[:, cur_L:cur_L+1] + time_embed[:, 0:1]
            
            if t != self.timesteps - 1:
                cur_to_prev_lidar_rt = camera_meta_datas_list[t+1]['curr_to_prev_lidar']

                curr_to_first_lidar = curr_to_first_lidar @ cur_to_prev_lidar_rt
                curr_to_first_lidar = curr_to_first_lidar.float()
                
            
            if vae_type != 0:
                img = vae.decode(summed_codes.squeeze(-3))
            else:
                img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

            img = (img + 1) / 2
            img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
            outputs.append(img.unsqueeze(1))
            
            if t == exit_step - 1:
                break
        if inference_mode:
            for b in self.unregistered_blocks: 
                b.sa.kv_caching(False)
                if self.use_temporal_attn:
                    b.sa_temporal.kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    module.sa.kv_caching(False)
                    if self.use_temporal_attn:
                        module.sa_temporal.kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        outputs_BVTCHW = torch.cat(outputs, dim = 1)
        return ret, idx_Bl_list, outputs_BVTCHW
    
    @for_visualize
    def vis_key_params(self, ep):
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=False, assign=False):
        for k in state_dict:
            if 'cfg_uncond' in k:
                old, new = state_dict[k], self.cfg_uncond.data
                min_tlen = min(old.shape[0], new.shape[0])
                if min_tlen == old.shape[0]:
                    state_dict[k] = torch.cat((old.to(device=new.device, dtype=new.dtype), new[min_tlen:]))
                else:
                    state_dict[k] = old[:min_tlen]
        
        for buf_name in ('lvl_1L', 'attn_bias_for_masking', 'Infinity_visible_kvlen', 'Infinity_invisible_qlen'):
            state_dict.pop(buf_name, None)
            if hasattr(self, buf_name):
                state_dict[buf_name] = getattr(self, buf_name)
        
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(
        self,
        aln_init: float,
        aln_gamma_init: float,
        scale_head: float,
        scale_proj: int,
    ):
        # init head's norm
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(aln_init)    # there's no gamma for head
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        # init head's proj
        if scale_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(scale_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(scale_head)
                self.head[-1].bias.data.zero_()
        
        depth = len(self.unregistered_blocks)
        for block_idx, sab in enumerate(self.unregistered_blocks):
            # init proj
            scale = 1 / math.sqrt(2*depth if scale_proj == 1 else 2*(1 + block_idx))
            if scale_proj == 1:
                sab.sa.proj.weight.data.mul_(scale)
                if self.use_temporal_attn:
                    sab.sa_temporal.proj.weight.data.mul_(scale)
                sab.ca.proj.weight.data.mul_(scale)
                sab.ffn.fc2.weight.data.mul_(scale)
            # if sab.using_swiglu:
            #     nn.init.ones_(sab.ffn.fcg.bias)
            #     nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            
            # init ada_lin
            if hasattr(sab, 'ada_lin'):
                lin = sab.ada_lin[-1]
                lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
                if hasattr(lin, 'bias') and lin.bias is not None:
                    lin.bias.data.zero_()
            if hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2, :].mul_(aln_gamma_init)  # init gamma
                sab.ada_gss.data[:, :, 2:, :].mul_(aln_init)        # init scale and shift
            if self.use_temporal_attn:
                if hasattr(sab, 'ada_lin_temporal'):
                    lin = sab.ada_lin_temporal[-1]
                    lin.weight.data[:self.C].mul_(aln_gamma_init)     # init gamma
                    lin.weight.data[self.C:].mul_(aln_init)           # init scale and shift
                    if hasattr(lin, 'bias') and lin.bias is not None:
                        lin.bias.data.zero_()
                if hasattr(sab, 'ada_gss_temporal'):
                    sab.ada_gss_temporal.data[:, :, :1, :].mul_(aln_gamma_init)  # init gamma
                    sab.ada_gss_temporal.data[:, :, 1:, :].mul_(aln_init)        # init scale and shift
                        
        lin = self.shared_ada_lin[1]
        if hasattr(lin, 'weight'):
            lin.weight.data[:2*self.C].mul_(aln_gamma_init)     # init gamma
            lin.weight.data[2*self.C:].mul_(aln_init)           # init scale and shift
            if hasattr(lin, 'bias') and lin.bias is not None:
                lin.bias.data.zero_()
        if self.use_temporal_attn:
            lin = self.shared_ada_lin_temporal[1]
            if hasattr(lin, 'weight'):
                lin.weight.data[:self.C].mul_(aln_gamma_init)     # init gamma
                lin.weight.data[self.C:].mul_(aln_init)           # init scale and shift
            if hasattr(lin, 'bias') and lin.bias is not None:
                lin.bias.data.zero_()
        # zero init
        if 'ray' in self.view_embed_type:
            zero_module(self.view_embed[-1])

        if self.time_embed_type == "fourier":
            zero_module(self.time_embedder[-1])

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str):
        raise NotImplementedError
    
    def get_last_shared_layer(self):
        return self.gpt.unregistered_blocks[-1].ffn.fc2


def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    # make sure logits_BlV is not nan
    logits_BlV = torch.where(torch.isnan(logits_BlV), torch.zeros_like(logits_BlV), logits_BlV)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def raynova_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

@register_model
def raynova_20b(depth=58, embed_dim=4608, num_heads=4608//128, drop_path_rate=0.25, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})

# model configuration for scaling Infinity transformer
@register_model
def raynova_layer12(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def raynova_layer16(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def raynova_layer24(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def raynova_layer32(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def raynova_layer40(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
@register_model
def raynova_layer48(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, **kwargs): 
    return RAYNOVA(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
