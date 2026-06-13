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

import random
import time
import gc
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os.path as osp
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2
import gc

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from infinity.utils.misc import get_scene_description, clip_cache, stop_gradient


Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


def grad_norm_weights(losses, shared_params):
    """
    losses: list of task losses, e.g. [loss1, loss2]
    shared_params: list of shared model parameters
    return: normalized weights [w1, w2]
    """
    grads = []
    for loss in losses:
        g = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=False)
        norm = torch.norm(torch.stack([p.norm() for p in g]))
        grads.append(norm + 1e-8)  # avoid div by 0

    inv_grads = [1.0 / g for g in grads]
    sum_inv = sum(inv_grads)
    weights = [inv_g / sum_inv for inv_g in inv_grads]
    return weights


class RayNovaTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: Infinity, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool, num_views: int, timesteps : int,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None, time_chunk=1,
    ):
        super(RayNovaTrainer, self).__init__()
        self.dbg_unused = dbg_unused
        
        self.zero = zero
        self.vae_type = vae_type
        self.num_views = num_views
        self.timesteps = timesteps
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')

        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device) # sum(pn ** 2) * num_views * timesteps
            cur = 0
            for timestep in range(self.timesteps):
                for raw_pn in raw_scale_schedule:
                    l = raw_pn*raw_pn*num_views
                    self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn*num_views) if self.eq_loss == 2 else raw_pn)
                    cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)

        self.time_chunk = time_chunk

    
    @torch.no_grad()
    def eval_ep(self, ep: int, args: arg_util.Args, ld_val: DataLoader, text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, last_step : int = 0):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.gpt.training
        self.gpt.eval()
        mvt_imgs_list = []
        camera_meta_datas_list = []
        caption_list = []
        for data in ld_val:
            mv_imgs = data['img_inputs'][0]
            meta_datas = data['img_metas'].data[0]
            batch_seq_ids = [item['sequence_group_idx'] for item in meta_datas]
            batch_seq_ids = torch.Tensor(batch_seq_ids).int()
            curr_to_prev_lidar = [item['curr_to_prev_lidar_rt'] for item in meta_datas]
            curr_to_prev_lidar = torch.stack(curr_to_prev_lidar, dim=0)
            rot, trans, intrins, post_rot, post_trans = data['img_inputs'][1:]
            camera_meta_datas = {
                'rot': rot,
                'trans': trans,
                'intrins': intrins,
                'post_rot': post_rot,
                'post_trans': post_trans,
                'seq_ids': batch_seq_ids,
                'curr_to_prev_lidar': curr_to_prev_lidar
            }
            if args.num_views == 1:
                mv_imgs = mv_imgs[:, 1:2]
                for key in camera_meta_datas:
                    if key not in ['seq_ids', 'curr_to_prev_lidar']:
                        camera_meta_datas[key] = camera_meta_datas[key][:, 1:2]

            for key in camera_meta_datas:
                camera_meta_datas[key] = camera_meta_datas[key].to(args.device, non_blocking=True) 
            camera_meta_datas['size'] = (mv_imgs.shape[4], mv_imgs.shape[3]) # (width, height)
            
            camera_meta_datas_list.append(camera_meta_datas)
            #TODO: remove unneccessary keys in meta_datas when appending
            caption_list.append(meta_datas)
            # get multi-frame multi-view images
            mvt_imgs_list.append(mv_imgs.unsqueeze(2)) 
            if len(mvt_imgs_list) < args.timesteps:
                continue
            # elif (camera_meta_datas_list[0]['seq_ids'] == camera_meta_datas_list[-1]['seq_ids']).sum() == 0:
            #     mvt_imgs_list.pop(0) # one-stride streaming
            #     camera_meta_datas_list.pop(0)
            #     continue
            else:
                inp_BVT3HW = torch.cat(mvt_imgs_list, dim=2) # BxVxTx3xHxW
                # mvt_imgs_list.clear() # non-overlap streaming
            assert len(camera_meta_datas_list) == len(mvt_imgs_list) == args.timesteps

            with torch.no_grad():   
                captions = get_scene_description(caption_list[0])
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.to(args.device, non_blocking=True)
                mask = tokens.attention_mask.to(args.device, non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
            
            kv_compact = []
            for len_i, feat_i in zip(lens, text_features.unbind(0)):
                kv_compact.append(feat_i[:len_i])
            kv_compact = torch.cat(kv_compact, dim=0)
            text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)
            
            
            inp_BVT3HW = inp_BVT3HW.to(args.device, non_blocking=True)
            V = self.vae_local.vocab_size
            num_views = inp_BVT3HW.shape[1]
            T = 1 if inp_BVT3HW.dim() == 5 else inp_BVT3HW.shape[2]
            B = inp_BVT3HW.shape[0]

            h_div_w = inp_BVT3HW.shape[-2] / inp_BVT3HW.shape[-1]
            h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            # scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
            scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]
            # [forward]
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in scale_schedule]
            # raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_BVT3HW.flatten(0,1), scale_schedule=vae_scale_schedule)

            # x_BmLsC_wo_prefix, gt_ms_idx_Bmfls = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_BVT3HW.flatten(0,1), raw_features, device)
            # x_BLC_wo_prefix: torch.Size([bs, 2*2*v+3*3*v+...+64*64*v, d or 4d])
            outputs = self.vae_local.encode(inp_BVT3HW.flatten(0,2), scale_schedule=vae_scale_schedule)
            gt_ms_idx_Bmls = outputs[3]
            x_BmCthw_wo_prefix = outputs[5]
            x_BmCLs_wo_prefix = [item.flatten(2,4) for item in x_BmCthw_wo_prefix]
            x_BmCLs_wo_prefix = torch.cat(x_BmCLs_wo_prefix, dim=2)
            x_BmLsC_wo_prefix = x_BmCLs_wo_prefix.permute(0, 2, 1) # B x L x C
            x_BVLC_wo_prefix = []
            gt_ms_idx_Bl = []
        
            x_BVLsC_wo_prefix = x_BmLsC_wo_prefix.reshape(B, self.num_views, self.timesteps, *x_BmLsC_wo_prefix.shape[1:])
            gt_ms_idx_Bmls = [item.flatten(1,3) for item in gt_ms_idx_Bmls] # B x L x C
            for tid in range(self.timesteps):
                for sid in range(len(gt_ms_idx_Bmls)):
                    l, c = gt_ms_idx_Bmls[sid].shape[1], gt_ms_idx_Bmls[sid].shape[2]
                    gt_ms_idx_bvtlc = gt_ms_idx_Bmls[sid].reshape(B, self.num_views, self.timesteps, l, c)
                    gt_ms_idx_Bl.append(gt_ms_idx_bvtlc[:, :, tid].reshape(B, self.num_views*l, c))

            for tid in range(self.timesteps):
                cur_l = 0
                for sid, (pt, ph, pw) in enumerate(scale_schedule[1:]):
                    x_BVLC_wo_prefix.append(x_BVLsC_wo_prefix[:, :, tid, cur_l:cur_l+ph*pw].flatten(1,2)) # B x l x c
                    cur_l += ph*pw
            
            x_BVLC_wo_prefix = torch.cat(x_BVLC_wo_prefix, dim=1)

            # truncate scales
            training_scales = args.always_training_scales
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum() * self.num_views * self.timesteps
            x_BVLC_wo_prefix = x_BVLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod() * self.num_views * self.timesteps), :]
            self.gpt.forward
            logits_BLV = self.gpt(text_cond_tuple, x_BVLC_wo_prefix, scale_schedule=scale_schedule[:training_scales], camera_meta_datas_list=camera_meta_datas_list)

            if training_scales < len(scale_schedule):
                raw_last_l = np.prod(scale_schedule[training_scales]) * num_views
            else:
                raw_last_l = np.prod(scale_schedule[-1]) * num_views
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            seq_mask = camera_meta_datas_list[0]['seq_ids'] == camera_meta_datas_list[-1]['seq_ids']

            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.val_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                loss = loss * seq_mask[:, None, None]

                cur_acc_sum = logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).argmax(dim=-1) == gt_BL
                cur_acc_sum = (cur_acc_sum * seq_mask[:, None, None]).sum() * seq_mask.shape[0] / (seq_mask.sum() + 1e-2)
                
                cur_acc_tail_sum = logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2)[:, -raw_last_l:].argmax(dim=-1) == gt_BL[:, -raw_last_l:]
                cur_acc_tail_sum = (cur_acc_tail_sum * seq_mask[:, None, None]).sum() * seq_mask.shape[0] / (seq_mask.sum() + 1e-2)
                
                acc_mean += cur_acc_sum * (100/(gt_BL.shape[1]*gt_BL.shape[2]))
                acc_tail += cur_acc_tail_sum * (100/(gt_BL.shape[1]*gt_BL.shape[2]))

                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1) * seq_mask.shape[0] / (torch.sum(seq_mask) + 1e-2)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                raise NotImplementedError(f'args.use_bit_label={args.use_bit_label}')
                loss = self.val_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)
                acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/self.logits_BLV.shape[1])
                acc_tail += (logits_BLV.data[:, -raw_last_l:].argmax(dim=-1) == gt_BL[:, -raw_last_l:]).sum() * (100/self.raw_last_l)

            mvt_imgs_list.pop(0) # one-stride streaming
            camera_meta_datas_list.pop(0)
            
            L_mean += torch.mean(loss) * B
            L_tail += torch.mean(loss[:, -raw_last_l:]) * B
            tot += B
            
        # TODO: save image
        # first_camera_meta_datas = {}
        # for k, v in camera_meta_datas.items():
        #     if k != 'size':
        #         first_camera_meta_datas[k] = v[0:1]
        gen_cam_meta_datas_list = [{} for _ in range(len(camera_meta_datas_list))]
        for i, camera_meta_datas in enumerate(camera_meta_datas_list):
            for ck, cv in camera_meta_datas.items():
                gen_cam_meta_datas_list[i][ck] = cv[0:1]
 
        # generated_image = gen_one_img_mvt(
        #     self.gpt_wo_ddp,
        #     self.vae_local,
        #     text_tokenizer,
        #     text_encoder,
        #     captions[0],
        #     gen_cam_meta_datas_list,
        #     g_seed=args.g_seed,
        #     gt_leak=0,
        #     gt_ls_Bl=None,
        #     cfg_list=3,
        #     tau_list=0.5,
        #     scale_schedule=scale_schedule,
        #     cfg_insertion_layer=[0],
        #     vae_type=args.vae_type,
        #     sampling_per_bits=1,
        #     enable_positive_prompt=0,
        # )
        # print(generated_image.shape)
        self.gpt.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()

        # wandb logging
        wandb_log_dict = {"Validation/L_mean": L_mean, 'Validation/Acc_mean': acc_mean, 'Validation/Acc_tail': acc_tail}
        wandb_utils.log(wandb_log_dict, step=last_step)
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def _get_action_cluster_weight(self, gt_future_traj, gt_action_id):
        # gt_future_traj: [B, T, L, 4]
        # gt_action_id: [B, T]
        
        B, T = gt_action_id.shape
        gt_future_traj = gt_future_traj[..., :2].to(self.action_centers.device)
        if self.action_centers.shape[1] != gt_future_traj.shape[2]:
            downsample = self.action_centers.shape[1] // gt_future_traj.shape[2]
            assert self.action_centers.shape[1] % gt_future_traj.shape[2] == 0
            assert downsample == 5
            action_centers = self.action_centers[:, downsample-1::downsample]
        else:
            action_centers = self.action_centers
        dist = torch.norm(gt_future_traj[:, :, None]-action_centers[None, None], p=2, dim=-1)
        dist = torch.mean(dist, dim=3) * 3
        weight = torch.clip(dist, min=0, max=100)
        weight = weight.flatten(0, 1)
        gt_action_id = gt_action_id.reshape(-1)
        ids = torch.arange(B*T).to(gt_action_id.device)
        weight[ids, gt_action_id] = 1
        weight = weight.reshape(B, T, weight.shape[-1])
        return weight

    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_BVT3HW: FTen, text_cond_tuple: Union[ITen, FTen], camera_meta_datas_list: list, bbox_sequence: list, action_cluster_ids: ITen, gt_future_trajectory: FTen, remove_casuality: bool,
        map_sequence: list, args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_BVT3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        num_views = inp_BVT3HW.shape[1]
        num_frames = inp_BVT3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_BVT3HW.device

        h_div_w = inp_BVT3HW.shape[-2] / inp_BVT3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        #TODO: figure the pacth T
        # scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]  
        scale_schedule = [ (1, h, w) for (t, h, w) in scale_schedule]      

        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in scale_schedule]
                    raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_BVT3HW.flatten(0,2), scale_schedule=vae_scale_schedule) # flatten B,V,T dimensions for parallel VAE encoding

            x_BmLsC_wo_prefix, gt_ms_idx_Bmfls = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_BVT3HW.flatten(0,2), raw_features, device)
            # x_BLC_wo_prefix: torch.Size([B*V*T, (2*2+3*3+...+64*64), d or 4d])
            # gt_ms_idx_Bmfls: list [B*V*T, 1*1, C], [B*V*T, 2*2, C] ... [B*V*T, 16*16, C]
            # print("x_BmLsC_wo_prefix", x_BmLsC_wo_prefix.shape)
            # for i, g in enumerate(gt_ms_idx_Bmfls):
            #     print(f"gt_ms_idx - level {i}", g.shape)
            x_BVLC_wo_prefix = []
            gt_ms_idx_Bl = []

            x_BVLsC_wo_prefix = x_BmLsC_wo_prefix.reshape(B, num_views, num_frames, *x_BmLsC_wo_prefix.shape[1:]) # B x V x T x L x c
            for tid in range(num_frames):
                for sid in range(len(gt_ms_idx_Bmfls)):
                    l, c = gt_ms_idx_Bmfls[sid].shape[1], gt_ms_idx_Bmfls[sid].shape[2]
                    gt_ms_idx_bvtlc = gt_ms_idx_Bmfls[sid].reshape(B, num_views, num_frames, l, c)
                    gt_ms_idx_Bl.append(gt_ms_idx_bvtlc[:, :, tid].reshape(B, num_views*l, c))
            
            for tid in range(num_frames):
                cur_l = 0
                for sid, (pt, ph, pw) in enumerate(scale_schedule[1:]):
                    x_BVLC_wo_prefix.append(x_BVLsC_wo_prefix[:, :, tid, cur_l:cur_l+ph*pw].flatten(1,2)) # B x l x c
                    cur_l += ph*pw
            
            x_BVLC_wo_prefix = torch.cat(x_BVLC_wo_prefix, dim=1) # B x L x c
            # truncate scales
            training_scales = args.always_training_scales

            # this is the total number of tokens in the training sequence
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum() * num_views * num_frames

            x_BVLC_wo_prefix = x_BVLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()*num_views*num_frames), :]

            logits_BLV = self.gpt(
                text_cond_tuple, x_BVLC_wo_prefix, 
                scale_schedule=scale_schedule[:training_scales], 
                camera_meta_datas_list=camera_meta_datas_list, 
                bbox_sequence=bbox_sequence, 
                map_sequence=map_sequence, 
                vae=self.vae_local
            ) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            seq_len = logits_BLV.shape[1]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]

            seq_mask = camera_meta_datas_list[0]['seq_ids'] == camera_meta_datas_list[-1]['seq_ids']

            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                loss = loss * seq_mask[:, None, None]
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1) * seq_mask.shape[0] / (torch.sum(seq_mask) + 1e-2)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                raise NotImplementedError(f'args.use_bit_label = {args.use_bit_label}')
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw * num_views)])
                lw = lw * num_frames
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / seq_len
            loss = loss.mul(lw).sum(dim=-1).mean()

        # [backward]
        total_loss = loss
        total_loss_value = total_loss.item()
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=total_loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)
            
        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1)
                res_loss = torch.sum(res_loss * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1)
            res_bit_acc = torch.sum(res_bit_acc * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float()
            res_token_acc = torch.sum(res_token_acc * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
                        
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            res_loss = res_loss.reshape(num_frames, -1)
            res_bit_acc = res_bit_acc.reshape(num_frames, -1)
            res_token_acc = res_token_acc.reshape(num_frames, -1)
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                #TODO: check the timestep
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod() * num_views
                L_list.append(res_loss[:, start:end].mean().item())
                acc_bit_list.append(res_bit_acc[:, start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[:, start:end].mean().item() * 100.)
                ptr = end
            
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)

            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            
            local_valid = 1 if seq_mask.any() else 0
            local_valid_t = torch.tensor([local_valid], device=seq_mask.device)
            tdist.all_reduce(local_valid_t, op=tdist.ReduceOp.MIN)  
            global_valid = local_valid_t.item()
            if global_valid:
                wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
                metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t)    # todo: Accm, Acct

                for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                    wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                    wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                    wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
                wandb_utils.log(wandb_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t

    def recurrent_train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_BVT3HW: FTen, text_cond_tuple: Union[ITen, FTen], camera_meta_datas_list: list, bbox_sequence: list, action_cluster_ids: ITen, gt_future_trajectory: FTen, remove_casuality: bool,
        map_sequence: list, args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_BVT3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        num_views = inp_BVT3HW.shape[1]
        V = self.vae_local.vocab_size
        device = inp_BVT3HW.device

        h_div_w = inp_BVT3HW.shape[-2] / inp_BVT3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        #TODO: figure the pacth T
        # scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]  
        scale_schedule = [ (1, h, w) for (t,h, w) in scale_schedule]      
                
        kv_compact, lens, cu_seqlens_k, max_seqlen = text_cond_tuple
        text_cond_tuples_list = []
        for tid in range(self.timesteps):
            kv_compact_frame = []
            lens_frame = []
            for bid in range(B):
                sid = cu_seqlens_k[bid*self.timesteps + tid]
                eid = cu_seqlens_k[bid*self.timesteps + tid + 1]
                kv_compact_frame.append(kv_compact[sid:eid])
                lens_frame.append(lens[bid*self.timesteps + tid])
            kv_compact_frame = torch.cat(kv_compact_frame, dim=0)
            
            cu_seqlens_k_frame = np.cumsum([0] + lens_frame, axis=0)
            cu_seqlens_k_frame = torch.tensor(cu_seqlens_k_frame, dtype=torch.int32, device=device)

            max_seqlen_frame = np.max(lens_frame)
            text_cond_tuples_list.append((kv_compact_frame, lens_frame, cu_seqlens_k_frame, max_seqlen_frame))
        
                
        # [forward]
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                if args.apply_spatial_patchify:
                    vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                else:
                    vae_scale_schedule = [(pt, ph, pw) for pt, ph, pw in scale_schedule]

        training_scales = args.always_training_scales
        
        if args.object_condition and np.random.rand() < self.gpt.object_cond_drop_rate:
            if args.bbox_img_coord:
                empty_frame_bbox_list = [[torch.zeros((0, 8, 6))] * B] * num_views
            else:
                empty_frame_bbox_list = [[torch.zeros((0, 8, 3))] * B] * num_views
            empty_frame_label_list = [[torch.zeros(0)] * B] * num_views
            empty_frame_bbox_num_list = [[0] * B] * num_views
            empty_frame_bbox_sequence = [empty_frame_bbox_list, empty_frame_label_list, empty_frame_bbox_num_list]
            
            if args.condition_rope:
                empty_frame_bbox_center_list = [[torch.zeros((0, 3))] * B] * num_views
                empty_frame_bbox_sequence.append(empty_frame_bbox_center_list)
            bbox_sequence = [empty_frame_bbox_sequence] * self.timesteps

        if args.map_condition and np.random.rand() < self.gpt.map_condition_drop_rate:
            if args.bbox_img_coord:
                empty_map_points_list = [[torch.zeros((0, args.map_sample_points_num, 6))] * B] * num_views
            else:
                empty_map_points_list = [[torch.zeros((0, args.map_sample_points_num, 3))] * B] * num_views
            empty_map_label_list = [[torch.zeros(0)] * B] * num_views
            empty_map_num_list = [[0] * B] * num_views
            empty_map_points_mask_list = [[torch.zeros((0, args.map_sample_points_num))] * B] * num_views
            
            
            empty_frame_map_sequence = [empty_map_points_list, empty_map_label_list, empty_map_num_list, empty_map_points_mask_list]
            if args.condition_rope:
                empty_map_center_list = [[torch.zeros((0, 3))] * B] * num_views
                empty_frame_map_sequence.append(empty_map_center_list)
            
            map_sequence = [empty_frame_map_sequence] * self.timesteps

        logits_BLV = []
        gt_BL = []
        cache = None
        for block_chunk_ in self.gpt.block_chunks:
            for module in block_chunk_.module.module:
                module.sa_temporal.recurrent_caching(True)

        for t in range(0, self.timesteps):
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    raw_features_frame, _, _ = self.vae_local.encode_for_raw_features(inp_BVT3HW[:, :, t].flatten(0,1), scale_schedule=vae_scale_schedule)  # (B*V, C, H, W)
            x_BmLsC_wo_prefix_frame, gt_ms_idx_Bmfls_frame = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_BVT3HW[:, :, t].flatten(0,1), raw_features_frame, device) # (B*V, L, C)
            x_BVLsC_wo_prefix_frame = x_BmLsC_wo_prefix_frame.reshape(B, num_views, *x_BmLsC_wo_prefix_frame.shape[1:]) # B x V x L x c

            x_BVLC_wo_prefix_frame = []
            gt_ms_idx_Bl_frame = []

            for sid in range(len(gt_ms_idx_Bmfls_frame)):
                l, c = gt_ms_idx_Bmfls_frame[sid].shape[1], gt_ms_idx_Bmfls_frame[sid].shape[2]
                gt_ms_idx_bvtlc_frame = gt_ms_idx_Bmfls_frame[sid].reshape(B, num_views, l, c)
                gt_ms_idx_Bl_frame.append(gt_ms_idx_bvtlc_frame.reshape(B, num_views*l, c))

            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum() * num_views
            gt_BLs = torch.cat(gt_ms_idx_Bl_frame, dim=1)[:, :training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            gt_BL.append(gt_BLs)

            cur_l = 0
            for sid, (pt, ph, pw) in enumerate(scale_schedule[1:]):
                x_BVLC_wo_prefix_frame.append(x_BVLsC_wo_prefix_frame[:, :, cur_l:cur_l+ph*pw].flatten(1,2)) # B x l x c
                cur_l += ph*pw

            x_BVLC_wo_prefix_frame = torch.cat(x_BVLC_wo_prefix_frame, dim=1) # B x L x c

            with self.gpt_opt.amp_ctx:
                if t > self.time_chunk:
                    input_camera_meta_datas_list = camera_meta_datas_list[t-self.time_chunk:t+1]
                else:
                    input_camera_meta_datas_list = camera_meta_datas_list[:t+1]

                
                output = self.gpt(
                    # text_cond_tuple, 
                    text_cond_tuples_list[t],
                    x_BVLC_wo_prefix_frame, 
                    scale_schedule=scale_schedule[:training_scales], 
                    camera_meta_datas_list=input_camera_meta_datas_list, 
                    bbox_sequence=bbox_sequence[t:t+1] if args.object_condition else None, 
                    vae=self.vae_local,
                    cache=cache,
                    map_sequence=map_sequence[t:t+1] if args.map_condition else None,
                ) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
                if isinstance(output, tuple):
                    logits_BLsV, cache = output
                else:
                    logits_BLsV = output
                logits_BLV.append(logits_BLsV.detach())
                seq_mask = camera_meta_datas_list[t]['seq_ids'] == camera_meta_datas_list[t]['seq_ids']
                seq_len = logits_BLsV.shape[1]            

                if args.use_bit_label:
                    tmp_bs, tmp_seq_len, tmp_channel = logits_BLsV.shape
                    loss = self.train_loss(logits_BLsV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BLs)
                    loss = loss * seq_mask[:, None, None]
                    if args.bitloss_type == 'mean':
                        loss = loss.mean(dim=-1) * seq_mask.shape[0] / (torch.sum(seq_mask) + 1e-2)
                    elif args.bitloss_type == 'sum':
                        loss = loss.sum(dim=-1)
                    else:
                        raise NotImplementedError(f'{args.bitloss_type=}')
                else:
                    raise NotImplementedError(f'args.use_bit_label = {args.use_bit_label}')

                if self.reweight_loss_by_scale:
                    lw = []
                    last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
                    for (pt, ph, pw) in scale_schedule[:training_scales]:
                        this_scale_area = np.sqrt(pt * ph * pw)
                        lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw * num_views)])
                    lw = torch.tensor(lw, device=loss.device)[None, ...]
                    lw = lw / lw.sum()
                else:
                    lw = 1. / seq_len
                
                loss = loss.mul(lw).sum(dim=-1).mean()
                
                loss = loss / self.timesteps
                # [backward]
                stepping_ = stepping and (t == self.timesteps - 1)
                grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping_, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)

        gt_BL = torch.cat(gt_BL, dim=1)

        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)
            
        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        logits_BLV = torch.cat(logits_BLV, dim=1)
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1)
                res_loss = torch.sum(res_loss * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1)
            res_bit_acc = torch.sum(res_bit_acc * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float()
            res_token_acc = torch.sum(res_token_acc * seq_mask[:, None], dim=0) / (seq_mask.sum() + 1e-2)
                        
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            res_loss = res_loss.reshape(self.timesteps, -1)
            res_bit_acc = res_bit_acc.reshape(self.timesteps, -1)
            res_token_acc = res_token_acc.reshape(self.timesteps, -1)
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                #TODO: check the timestep
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod() * num_views
                L_list.append(res_loss[:, start:end].mean().item())
                acc_bit_list.append(res_bit_acc[:, start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[:, start:end].mean().item() * 100.)
                ptr = end
                        
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)

            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
                        
            local_valid = 1 if seq_mask.any() else 0
            local_valid_t = torch.tensor([local_valid], device=seq_mask.device)
            tdist.all_reduce(local_valid_t, op=tdist.ReduceOp.MIN)  
            global_valid = local_valid_t.item()
            if global_valid:
                wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
                metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t)    # todo: Accm, Acct

                for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                    wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                    wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                    wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
                wandb_utils.log(wandb_log_dict, step=g_it)

        return grad_norm_t, scale_log2_t
    
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(RayNovaTrainer, self).__repr__().replace(RayNovaTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
