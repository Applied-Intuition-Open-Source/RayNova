import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/core_stack'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party/core_stack/onroad'))


import gc
import json
import math
# os.environ['AWS_ACCESS_KEY_ID'] = 'cd0146b0fd5c24625a928b242d19f7e0dec18424'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'HN7mDT0pooo+3E40lyab8rrNfIied/33pCbpyrSEDuA='
# os.environ['URSA_SDK_GRPC_HOSTNAME'] = 'grpc.neuron.oci.applied.dev'
# os.environ['AWS_DEFAULT_REGION'] = 'us-phoenix-1'
# os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
# os.environ['AWS_DEFAULT_REGION'] = 'us-chicago-1'
# os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com'

import random
import time
import traceback
from collections import deque
from contextlib import nullcontext
from functools import partial
from distutils.util import strtobool
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import builtins

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import transformers
import torch.distributed as tdist
import wandb
import infinity.utils.dist as dist
from infinity.utils.save_and_load import CKPTSaver, auto_resume
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from infinity.utils.misc import get_scene_description, project_corners_to_views
from infinity.utils.data_sampler import InfiniteGroupEachSampleInBatchSampler, FiniteGroupEachSampleInBatchSampler
from infinity.models.ema import get_ema_model

from infinity.utils.s3_file_utils import load_bytes_file, download_s3_folder, save_state_dict_to_s3, save_yaml_to_s3, download_s3_file
from nuscenes_tools.nuscenes_utils.collate import collate as collate_nuscenes

from scenarionet_tools.collate import collate as collate_nuplan
from scenarionet_tools.scenarionet_data_wrapper import convert_bbox
from scenarionet_tools.scenarionet_data_wrapper import build_scenarionet_dataset
from scenarionet_tools.map_utils import convert_points
from scenarionet_tools.mixed_dataloader import ProbabilisticDataLoader
from nuscenes_tools.nuscenes_utils.box3d_instance import LiDARInstance3DBoxes

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
import warnings
# ignore all warnings
warnings.filterwarnings("ignore")
enable_timeline_sdk = False

def freeze_pretrained_params(model):
    # scale level embedding
    frozen_num = 0
    count_params = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    for name, param in model.lvl_embed.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.lvl_embed)
    for name, param in model.word_embed.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.word_embed)
    for name, param in model.head.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.head)
    for name, param in model.norm0_ve.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.norm0_ve)
    for name, param in model.text_proj_for_sos.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.text_proj_for_sos)
    for name, param in model.shared_ada_lin.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.shared_ada_lin)
    for name, param in model.head.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.head)
    for name, param in model.head_nm.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.head_nm)    
    for name, param in model.text_proj_for_ca.named_parameters():
        param.requires_grad = False
    frozen_num += count_params(model.text_proj_for_ca)    
    
    for block in model.unregistered_blocks:
        for name, param in block.sa.named_parameters():
            param.requires_grad = False
        for name, param in block.ffn.named_parameters():
            param.requires_grad = False
        block.ada_gss.requires_grad = False
        frozen_num += count_params(block.sa) + count_params(block.ffn) + block.ada_gss.numel() / 1e6
    
    print(f'[freeze_pretrained_params] frozen {round(frozen_num, 2)} M params')

    
    return model

def remove_casual(data):
    bs = len(data[0]['img_metas'].data)
    timesteps = len(data)
    curr_to_prev_lidar_rt_list = [[] for _ in range(bs)]
    
    curr_to_first_lidar_rt_list = [torch.stack([torch.eye(4)]*bs, dim=0)]
    for tid in range(1, timesteps):
        curr_to_prev_lidar_rt_list = []
        for bid in range(bs):
            curr_to_prev_rt = data[tid]['img_metas'].data[bid][0]['curr_to_prev_lidar_rt']
            curr_to_prev_lidar_rt_list.append(curr_to_prev_rt)
        curr_to_prev_lidar_rt = torch.stack(curr_to_prev_lidar_rt_list, dim=0)
        curr_to_first_lidar_rt_list.append(curr_to_first_lidar_rt_list[-1] @ curr_to_prev_lidar_rt)
    
    curr_to_first_lidar_rt = torch.stack(curr_to_first_lidar_rt_list, dim=1)
    
    shuffle_ids = np.arange(timesteps)
    np.random.shuffle(shuffle_ids[1:])
    
    new_data = []
    for i in range(timesteps):
        new_data.append(data[shuffle_ids[i]])
    
    new_curr_to_prev_lidar_rt_list = [torch.stack([torch.eye(4)]*bs, dim=0)]
    for i in range(1, timesteps):
        curr_id = shuffle_ids[i]
        prev_id = shuffle_ids[i-1]
        new_curr_to_prev_lidar_rt_list.append(torch.inverse(curr_to_first_lidar_rt[:, prev_id]) @ curr_to_first_lidar_rt[:, curr_id])
    
    new_curr_to_prev_lidar_rt = torch.stack(new_curr_to_prev_lidar_rt_list, dim=1)

    for tid in range(timesteps):
        for bid in range(bs):
            new_data[tid]['img_metas'].data[bid][0]['curr_to_prev_lidar_rt'] = new_curr_to_prev_lidar_rt[bid, tid]
    return new_data

def worker_init_fn(worker_id):
    torch.set_num_threads(1)  # Reduce thread contention

def build_everything_from_args(args: arg_util.Args, saver):
    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: # check the randomness
        misc.check_randomness(args)

    # build data
    iters_train, iters_val, ld_train, ld_val = build_nuplan_dataloaders(args)
    # train_h_div_w_list = list(ld_train.dataset.h_div_w_template2generator.keys())
    # print(f"{train_h_div_w_list=}")
    args.train_h_div_w_list = None

    # load VAE
    print(f'Load vae from {args.vae_ckpt}')
    if not os.path.exists(args.vae_ckpt):
        vae_ckpt = {}
    else:
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(args, vae_ckpt)
    
    # IMPORTANT: import heavy package `InfinityTrainer` after the Dataloader object creation/iteration to avoid OOM
    from trainer_raynova import RayNovaTrainer
    # build trainer
    trainer = RayNovaTrainer(
        is_visualizer=dist.is_visualizer(), device=args.device, raw_scale_schedule=args.scale_schedule, resos=args.resos,
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=args.tema, max_it=iters_train * args.ep,
        gpt_opt=gpt_optim, label_smooth=args.ls, z_loss_ratio=args.lz, eq_loss=args.eq, xen=args.xen,
        dbg_unused=args.dbg, zero=args.zero, vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, num_views=args.num_views, 
        timesteps=args.timesteps, time_chunk=args.time_chunk, other_args=args, 
    )
    
    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    for info in auto_resume_info:
        print(info)
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.dump_log()
    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again

    if (start_it != 0) and (start_it % iters_train == 0):
        start_ep += 1
    start_it = start_it % iters_train
    print(f"{start_it=}, {iters_train=}")
    
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, iters_val, ld_train, ld_val
    )


def build_model_optimizer(args, vae_ckpt):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.raynova_models.raynova import RAYNOVA, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params, freeze_params
    from infinity.utils.load import build_vae_raynova
    
    if args.online_t5:
        print(f'Loading T5 from {args.t5_path}...', flush=True)
        local_t5_path = args.t5_path
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(local_t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(local_t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        [p.requires_grad_(False) for p in text_encoder.parameters()]
    else:
        text_tokenizer = text_encoder = None
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp, gpt_wo_ddp_ema = build_vae_raynova(args, vae_ckpt, skip_gpt=False, device=args.model_init_device)
    del vae_ckpt
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    
    init_weights(gpt_wo_ddp, other_std=args.tini)
    gpt_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)

    if args.object_condition:
        if gpt_wo_ddp.bbox_encoder.use_text_encoder_init:
            assert text_tokenizer is not None and text_encoder is not None
            gpt_wo_ddp.bbox_encoder.prepare(text_tokenizer, text_encoder, args.class_names)

    if args.map_condition:
        if gpt_wo_ddp.map_encoder.use_text_encoder_init:
            assert text_tokenizer is not None and text_encoder is not None
            gpt_wo_ddp.map_encoder.prepare(text_tokenizer, text_encoder, args.map_names)
            
    if args.pretrained_ckpt is not None and (args.resume_path is None or args.resume_path==''):
        print(f"{args.pretrained_ckpt=}", flush=True)
        ckpt_state_dict = torch.load(load_bytes_file(args.pretrained_ckpt), map_location='cpu')
        ckpt_state_dict['pos_1LsC_start'] = ckpt_state_dict['pos_start']
        del ckpt_state_dict['pos_start']

        missing_keys, unexpected_keys = gpt_wo_ddp.load_state_dict(ckpt_state_dict, strict=False)
        print("Missing keys:", missing_keys, flush=True)
        print("Unexpected keys:", unexpected_keys, flush=True)
    
    if gpt_wo_ddp_ema is not None:
        del gpt_wo_ddp_ema
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    
    if args.rush_resume:
        print(f"{args.rush_resume=}", flush=True)
        cpu_d = torch.load(load_bytes_file(args.rush_resume), 'cpu')
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            # del state_dict['time_embed'] # only when increasing the timesteps
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            # del state_dict['time_embed'] # only when increasing the timesteps
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != gpt_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != gpt_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            if 'bbox_encoder._class_tokens' in state_dict and (state_dict['bbox_encoder._class_tokens'].shape[0] != gpt_wo_ddp.bbox_encoder._class_tokens.shape[0]):
                del state_dict['bbox_encoder._class_tokens']
            if state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape:
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        
        missing_keys, unexpected_keys = gpt_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
        print("Missing keys for rush resume:", missing_keys, flush=True)
        print("Unexpected keys for rush resume:", unexpected_keys, flush=True)
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)

    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('GPT', gpt_wo_ddp),
    )]) + '\n\n')

    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    if args.freeze_backbone:
        gpt_wo_ddp = freeze_pretrained_params(gpt_wo_ddp)

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        # Modified section to handle hybrid sharding without device_mesh
        if args.enable_hybrid_shard:
            # Log warning about hybrid shard compatibility
            print("WARNING: Hybrid sharding with device_mesh is not supported in torch 2.1.1")
            print("Falling back to regular sharding strategy")
            
            # Use regular sharding strategy instead
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
        
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)
        
        local_rank = dist.get_local_rank()
        device = f"cuda:{local_rank}"
        gpt_wo_ddp = gpt_wo_ddp.to(device)
        # Initialize FSDP without device_mesh parameter
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=local_rank,
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
        )#.to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=args.fsdp_orig, 
                sync_module_states=True, 
                limit_all_gathers=True,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
            
            # added for world model
            'st_ada_gss', 'temporal_ada_gss', 'spatial_gate', 'temporal_gate',
            'time_embed', 'pos_1LsC_start', 'null_pos_feature', 'null_class_feature',
            'ada_gss_temporal',

            # added for action prediction
            'action_ada_gss', 'action_start', 'action_pos', 'command_embedding'
            
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    
    names, paras, para_groups = filter_params(gpt_ddp if args.zero else gpt_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.tclip, args.zero)
    del names, paras, para_groups
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def build_nuplan_dataloaders(args: arg_util.Args, source_id=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.task_type == 't2i':
        dataset_train = build_scenarionet_dataset(args, rank, world_size, source_id, dataset_name='nuplan')
    else:
        raise NotImplementedError(f'args.task_type={args.task_type} not supported')
    vbs = round(args.batch_size)
    print(f"{args.batch_size=}, {vbs=}", flush=True)

    ld_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False, generator=args.get_different_generator_for_each_rank(), 
                          collate_fn=partial(collate_nuplan, samples_per_gpu=args.batch_size), worker_init_fn=worker_init_fn, drop_last=True,
                          multiprocessing_context="spawn")
    iters_train = 10101 // (args.batch_size * dist.get_world_size())
    if args.recurrent_training:
        iters_train = iters_train // (args.timesteps // args.time_chunk * 2)
    # iters_train = 2500 // (args.batch_size * dist.get_world_size())
    
    ld_val = None
    iters_val = 0
    print('training:')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}')

    print('validation: None')
    ld_train = iter(ld_train)
    del dataset_train
    return iters_train, iters_val, ld_train, ld_val


def main_train(config=None, experiment_tracker=None):

    if not dist.is_master():
        builtins.print = lambda *args, **kwargs: None
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    ret = build_everything_from_args(args, saver)
    
    if ret is None:
        return
    
    (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, iters_val, ld_train, ld_val
    ) = ret
    gc.collect(), torch.cuda.empty_cache()
    
    # import heavy packages after Dataloader object creation
    from trainer_raynova import RayNovaTrainer
    ret: Tuple[
        misc.TensorboardLogger, T5TokenizerFast, T5EncoderModel, RayNovaTrainer,
        int, int, str, List[Tuple[float, float]], Optional[int], Optional[DataLoader], DataLoader,
    ]

    world_size = int(os.environ["WORLD_SIZE"])
    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean, last_val_acc_mean, best_val_acc_mean = 999., 999., 0., 0.
    last_val_loss_tail, best_val_loss_tail, last_val_acc_tail, best_val_acc_tail = 999., 999., 0., 0.
    seg5 = np.linspace(1, args.ep, 5+1, dtype=int).tolist()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    milestone_ep_feishu_log = set(seg5[:])
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= args.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= args.ep:
            vis_milestone_ep.add(x)
    
    PARA_EMB, PARA_ALN, PARA_OT = 0, 0, 0
    for n, p in trainer.gpt_wo_ddp.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('class_emb', 'pos_1LC', 'lvl_embed')):
            PARA_EMB += p.numel()
        elif any(k in n for k in ('ada_lin',)):
            PARA_ALN += p.numel()
        else:
            PARA_OT += p.numel()
    PARA_ALL = PARA_EMB + PARA_ALN + PARA_OT
    
    # trainer.gpt_opt.log_param(ep=-1)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.ep // 10) if args.ep <= 100 else max(1, args.ep // 20)
    
    # ============================================= epoch loop begins =============================================
    L_mean, L_tail = -1, -1
    epochs_loss_nan = 0
    # build wandb logger
    if dist.is_master():
        if args.resume_wandb_name is not None:
            wandb_name = args.resume_wandb_name
            wandb.init(entity="research-interns",project="Infinity_stream", name=wandb_name, id=wandb_name, resume="must", reinit=True)
        else:
            wandb_utils.initialize(args)


    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info]  from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},    =======>  bed: {args.bed}  <=======\n')
        # set epoch for dataloader
        # if args.use_streaming_dataset:
        #     ld_train.dataset.set_epoch(ep)
        # last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val, text_tokenizer=text_tokenizer, text_encoder=text_encoder, last_step=0)
        # exit(0)
        # [train one epoch]
        stats, (sec, remain_time, finish_time), last_step = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=ld_train,
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            enable_timeline_sdk=enable_timeline_sdk,
        )
        
        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)
        
        # [check nan]
        epochs_loss_nan += int(not math.isfinite(L_mean))
        if (args.fp16 == 1 and epochs_loss_nan >= 2) or (args.fp16 != 1 and epochs_loss_nan >= 1):
            print(f'[rk{dist.get_rank():02d}] L_mean is {L_mean}, stopping training!', flush=True)
            sys.exit(666)
        
        # [logging]
        args.cur_phase = 'AR'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        args.last_Lnll, args.last_Ld, args.acc_all, args.acc_real, args.acc_fake, args.last_wei_g = min_L_mean, min_L_tail, None, (None if max_acc_mean < 0 else max_acc_mean), (None if max_acc_tail < 0 else max_acc_tail), grad_norm
        if math.isfinite(args.last_wei_g) and args.last_wei_g > 4:
            args.grad_boom = 'boom'
        
        AR_ep_loss = {}
        is_val_and_also_saving = ep == 0 or (ep + 1) % 5 == 0 or (ep + 1) == args.ep
        
        
        # if (ep + 1) < 10:
        #     law_stats = {
        #         'last_Lm': L_mean, 'best_Lm': min_L_mean, 'last_Am': acc_mean, 'best_Am': max_acc_mean,
        #         'last_Lt': L_tail, 'best_Lt': min_L_tail, 'last_At': acc_tail, 'best_At': max_acc_tail,
        #         'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
        #     }
        # elif is_val_and_also_saving:
        if is_val_and_also_saving:
            if ld_val is None or isinstance(ld_val, int):    # args.nodata or args.nova
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = 0.666, 0.555, 5.55, 6.66, 50000, 0.001
            else:
                # last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = 0.666, 0.555, 5.55, 6.66, 50000, 0.001
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val, text_tokenizer=text_tokenizer, text_encoder=text_encoder, last_step=last_step)
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, last_val_loss_mean), min(best_val_loss_tail, last_val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, last_val_acc_mean), max(best_val_acc_tail, last_val_acc_tail)
            AR_ep_loss['vL_mean'], AR_ep_loss['vL_tail'], AR_ep_loss['vacc_mean'], AR_ep_loss['vacc_tail'] = last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail
            print(f'  [*] [ep{ep}]  VAL {tot}  |  Lm: {last_val_loss_mean:.4f}, Lt: {last_val_loss_tail:.4f}, Accm: {last_val_acc_mean:.2f}, Acct: {last_val_acc_tail:.2f}, cost: {cost:.2f}s')
            law_stats = {
                'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        else: law_stats = None
        if dist.is_master() and law_stats is not None:
            stat_file = os.path.join(args.bed, 'law.stat')
            if os.path.exists(stat_file):
                with open(stat_file, 'r', encoding='utf-8') as law_fp: tag_to_epv = json.load(law_fp)
            else:
                tag_to_epv = {tag: {} for tag in law_stats.keys()}
            for tag, v in law_stats.items():
                tag_to_epv[tag][ep + 1] = v
            with open(stat_file, 'w', encoding='utf-8') as law_fp: json.dump(tag_to_epv, law_fp, indent=2)
            
            # ============= LEGACY =============
            with open(os.path.join(args.bed, 'law'), 'w') as law_fp:
                json.dump({
                    'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                    'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                    'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
                }, law_fp, indent=2)
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} ({L_mean:.3f}), Ltail {min_L_tail:.3f} ({L_tail:.3f}),  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        AR_ep_loss['L_mean'], AR_ep_loss['L_tail'], AR_ep_loss['acc_mean'], AR_ep_loss['acc_tail'] = L_mean, L_tail, acc_mean, acc_tail        
        args.dump_log()
        # exit(0)
    # ============================================= epoch loop ends =============================================
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats, iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    args.cur_phase = 'OK'
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    dist.barrier()
    return

g_speed_ls = deque(maxlen=128)
def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone, enable_timeline_sdk: bool,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    from trainer_raynova import RayNovaTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: RayNovaTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=20, verbose=True) as telling_dont_kill:
        last_touch = time.time()
        g_it, max_it = ep * iters_train, args.ep * iters_train
        
        doing_profiling = args.prof and ep == 0 and (args.profall or dist.is_master())
        maybe_record_function = record_function if doing_profiling else nullcontext
        trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
        
        last_t_perf = time.time()
        speed_ls: deque = g_speed_ls
        FREQ = min(args.prof_freq, 1)
        NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
        ranges = set([2 ** i for i in range(20)])
        if ep <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
        PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

        # with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
        #     local_ckpt_path = saver.sav(args=args, g_it=(g_it+1), next_ep=1, next_it=1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
        # exit(0)
        
        me = misc.MetricLogger()
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]

        # ============================================= iteration loop begins =============================================
        camera_meta_datas_list = []
        caption_list = []
        for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
            g_it = ep * iters_train + it

            # calling inc_step to sync the global_step
            # if enable_timeline_sdk:
            #     ndtimeline.inc_step()

            if (it+1) % FREQ == 0:
                speed_ls.append((time.time() - last_t_perf) / FREQ)
                last_t_perf = time.time()

                # if enable_timeline_sdk:
                #     ndtimeline.flush()
            # print("it outside", it)
            
            # Whether to shuffle the order of frames (except 1st frame)
            if np.random.random() < args.random_remove_casuality:
                remove_casuality = True
                data = remove_casual(data)
            else:
                remove_casuality = False
            
            with maybe_record_function('before_train'):
                # [get data]
                mv_imgs = [item['img_inputs'][0] for item in data] # [(B, V, 3, H, W)] * T
                inp = torch.stack(mv_imgs, dim=2)  # (B, V, T, 3, H, W)
                

                # Drop views randomly.
                # If args.random_drop_view is True, the num of views is randomly selected between [1, num_views]
                orig_num_views = inp.shape[1]
                if args.random_drop_view:
                    cur_num_views = np.random.randint(1, min(orig_num_views, args.num_views)+1)
                    
                    if args.adaptive_horizon:
                        num_frames = args.num_views * args.timesteps // cur_num_views
                        num_frames = min(num_frames, args.max_horizon)
                    else:
                        num_frames = args.timesteps
                else:
                    cur_num_views = min(args.num_views, orig_num_views)
                    num_frames = args.timesteps
                selected_views = np.random.choice(orig_num_views, cur_num_views, replace=False)
                selected_views = np.sort(selected_views)
                
                selected_frames_start = np.random.randint(0, args.max_horizon - num_frames + 1)
                selected_frames = np.arange(selected_frames_start, selected_frames_start + num_frames)
                
                inp = inp[:, selected_views]
                inp = inp[:, :, selected_frames]
                assert selected_frames_start + num_frames <= len(data)
                data = data[selected_frames_start:selected_frames_start + num_frames]
                

                meta_datas_list = [] # [dict()] * T
                for item in data:
                    batch_meta_datas = item['img_metas'].data
                    batch_meta_datas = [sample_meta_datas[0] for sample_meta_datas in batch_meta_datas]
                    meta_datas_list.append(batch_meta_datas)
                    
                caption_list = meta_datas_list
                batch_seq_ids = torch.zeros(inp.shape[0]).int()  # No need to check sequence validation
                                    
                num_views = inp.shape[1]
                batch_size = inp.shape[0]
                
                camera_meta_datas_list = []
                curr_to_first_lidar = torch.eye(4)[None].repeat(batch_size, 1, 1)
                for tid in range(num_frames):
                    curr_to_prev_lidar = [item['curr_to_prev_lidar_rt'] for item in meta_datas_list[tid]]
                    curr_to_prev_lidar = torch.stack(curr_to_prev_lidar, dim=0)
                    timesteps = [item['timestep'] for item in meta_datas_list[tid]]
                    timesteps = torch.Tensor(timesteps)
                    timesteps = timesteps * (args.timesteps - 1)
                    
                    if tid != 0:
                        curr_to_first_lidar = curr_to_first_lidar @ curr_to_prev_lidar
                    
                    # Prepare the camera parameters
                    rot, trans, intrins, post_rot, post_trans = data[tid]['img_inputs'][1:]
                    camera_meta_datas = {
                        'rot': rot, # [B, num_views, 3, 3] Camera extrinsics R
                        'trans': trans,  # [B, num_views, 3] Camera extrinsics t
                        'intrins': intrins,   # [B, num_views, 3, 3] camera intrinsic matrix
                        'post_rot': post_rot,  # [B, num_views, 3, 3] rotation (not actully used) and resize of each image"
                        'post_trans': post_trans,   # [B, num_views, 3] translation of each image
                        'seq_ids': batch_seq_ids,  # [B,] No need to check so set it to zero
                        'curr_to_prev_lidar': curr_to_prev_lidar,  # [B, 4, 4] current LiDAR to last sampled frame LiDAR coordinate RT matrix
                        'curr_to_first_lidar': curr_to_first_lidar, # [B, 4, 4] current LiDAR to first sampled frame LiDAR coordinate RT matrix
                        'timestep': timesteps,  # [B, ] normalized timestep w.r.t. max length (3s in default), the first frame is 0
                    }

                    for key in camera_meta_datas:
                        if key not in ['seq_ids', 'curr_to_prev_lidar', 'curr_to_first_lidar', 'timestep']:
                            camera_meta_datas[key] = camera_meta_datas[key][:, selected_views]           

                    camera_meta_datas['size'] = (inp.shape[-1], inp.shape[-2]) # (width, height)
                
                    camera_meta_datas_list.append(camera_meta_datas)
                                    
                # Convert bounding box to the required format
                sample_meta_data = {}
                if args.object_condition:
                    bbox_sequence = []
                    curr_to_first = torch.eye(4)[None].repeat(batch_size, 1, 1)
                    for tid in range(num_frames):
                        if tid != 0:
                            cur_to_prev = camera_meta_datas_list[tid]['curr_to_prev_lidar']
                            curr_to_first = curr_to_first @ cur_to_prev
                        frame_data = data[tid]
                        gt_bboxes_3d = frame_data['gt_bboxes_3d'].data
                        gt_labels_3d = frame_data['gt_labels_3d'].data
                        frame_bbox_list = [[] for _ in range(num_views)]
                        frame_label_list = [[] for _ in range(num_views)]
                        frame_bbox_num_list = [[] for _ in range(num_views)]
                        if args.condition_rope:
                            frame_bbox_center_list = [[] for _ in range(num_views)]
                        for bid in range(len(gt_bboxes_3d)):
                            frame_boxes_3d = gt_bboxes_3d[bid][0]
                            frame_labels = gt_labels_3d[bid][0]

                            if len(frame_labels) > 0:

                                corners = frame_boxes_3d.corners
                                center = frame_boxes_3d.gravity_center[:, None]
                                # For ablation study, otherwise box corners in the coordinate of first frame
                                if args.use_frame_coordinate:
                                    xyz = frame_boxes_3d.bottom_center
                                    dims = frame_boxes_3d.dims
                                    yaw = frame_boxes_3d.yaw

                                    # Convert boxes from the first frame coordinate to current frame coordinate
                                    xyz, yaw = convert_bbox(xyz, yaw, torch.inverse(curr_to_first[bid])[:3,:3], torch.inverse(curr_to_first[bid])[:3,3])
                                    boxes = torch.cat([xyz, dims, yaw[:, None]], dim=-1)
                                    frame_boxes_3d = LiDARInstance3DBoxes(boxes)            

                                sample_meta_data = {}
                                for key in ['rot', 'trans', 'intrins', 'post_rot', 'post_trans']:
                                    sample_meta_data[key] = camera_meta_datas_list[tid][key][bid]
                                sample_meta_data['size'] = camera_meta_datas_list[tid]['size']
                                # Check whether each box can be projected to each camera view
                                # box_coord_img: projected coordinate in camera space (x,y,d)
                                box_mask, box_coord_img = project_corners_to_views(corners, torch.inverse(curr_to_first)[bid], sample_meta_data, return_coord_2d=True)  # [N, V]
                                if args.condition_rope:
                                    _, bbox_center_img = project_corners_to_views(center, torch.inverse(curr_to_first)[bid], sample_meta_data, return_coord_2d=True)
                                    bbox_center_img = bbox_center_img.squeeze(1)
                            else:
                                box_mask = torch.zeros([1, num_views])
                            
                            for vid in range(num_views):
                                box_view_mask = box_mask[:, vid] > 0  # [N, ]
                                box_view_num = int(box_view_mask.sum())
                                if box_view_num > 0:
                                    view_boxes_3d = frame_boxes_3d[box_view_mask]
                                    view_corners = view_boxes_3d.corners
                                    view_labels = frame_labels[box_view_mask]
                                    view_corners_img = box_coord_img[box_view_mask, :, vid]

                                    if args.bbox_img_coord:
                                        view_corners = torch.cat([view_corners, view_corners_img], dim=-1)
                                    
                                    # view_corners = torch.cat([view_corners]*5, dim=0)
                                    # view_labels = torch.cat([view_labels]*5, dim=0)
                                    # box_view_num = box_view_num * 5
    
                                    frame_bbox_list[vid].append(view_corners)
                                    frame_label_list[vid].append(view_labels)
                                    frame_bbox_num_list[vid].append(box_view_num)
                                    
                                    if args.condition_rope:
                                        view_center_img = bbox_center_img[box_view_mask, vid]
                                        frame_bbox_center_list[vid].append(view_center_img)                     
                                else:
                                    if args.bbox_img_coord:
                                        frame_bbox_list[vid].append(torch.zeros((0, 8, 6)))
                                    else:
                                        frame_bbox_list[vid].append(torch.zeros((0, 8, 3)))
                                    frame_label_list[vid].append(torch.zeros(0))
                                    frame_bbox_num_list[vid].append(0)
                                    if args.condition_rope:
                                        frame_bbox_center_list[vid].append(torch.zeros((0, 3)))
                        if args.condition_rope:
                            bbox_sequence.append([frame_bbox_list, frame_label_list, frame_bbox_num_list, frame_bbox_center_list])
                        else:
                            bbox_sequence.append([frame_bbox_list, frame_label_list, frame_bbox_num_list])
                else:
                    bbox_sequence = None
                
                if args.map_condition:
                    map_sequence = []
                    curr_to_first = torch.eye(4)[None].repeat(batch_size, 1, 1)
                    for tid in range(num_frames):
                        if tid != 0:
                            cur_to_prev = camera_meta_datas_list[tid]['curr_to_prev_lidar']
                            curr_to_first = curr_to_first @ cur_to_prev
                        frame_data = data[tid]
                        map_labels = frame_data['map_type_labels'].data
                        map_points = frame_data['map_sampled_points'].data

                        frame_points_list = [[] for _ in range(num_views)]
                        frame_label_list = [[] for _ in range(num_views)]
                        frame_map_num_list = [[] for _ in range(num_views)]
                        frame_points_mask_list = [[] for _ in range(num_views)]
                        if args.condition_rope:
                            frame_map_center_list = [[] for _ in range(num_views)]
                        for bid in range(len(map_labels)):
                            frame_map_points = map_points[bid][0].to(torch.float32)
                            frame_map_labels = map_labels[bid][0]

                            if len(frame_map_labels) > 0:
                                if len(sample_meta_data) == 0:
                                    for key in ['rot', 'trans', 'intrins', 'post_rot', 'post_trans']:
                                        sample_meta_data[key] = camera_meta_datas_list[tid][key][bid]
                                    sample_meta_data['size'] = camera_meta_datas_list[tid]['size']
 
                                point_mask, point_coord_img = project_corners_to_views(frame_map_points, torch.inverse(curr_to_first)[bid], sample_meta_data, return_coord_2d=True, pointwise_mask=True)  # [N, V]
                                # point_mask: [N, n_points, V]
                                # point_coord_img: [N, n_points, V, 3]                                
                                if args.condition_rope:
                                    center_point = torch.sum(frame_map_points[:, :, None] * point_mask[..., None], dim=1) / (torch.sum(point_mask[..., None], dim=1) + 1e-6)
                                    # center_point: [N, V, 3]
                                    _, point_center_img = project_corners_to_views(center_point, torch.inverse(curr_to_first)[bid], sample_meta_data, return_coord_2d=True, pointwise_mask=True)
                                    # point_center_img: [N, V, V, 3]

                                    
                                # For ablation study, otherwise box corners in the coordinate of first frame
                                if args.use_frame_coordinate:
                                    # Convert points from the first frame coordinate to current frame coordinate
                                    frame_map_points = convert_points(frame_map_points, torch.inverse(curr_to_first[bid])[:3,:3], torch.inverse(curr_to_first[bid])[:3,3])                           

                            else:
                                point_mask = torch.zeros([1, args.map_sample_points_num, num_views])
                            
                            for vid in range(num_views):
                                if len(point_mask.shape) < 3:
                                    print(f"point_mask.shape: {point_mask.shape}")
                                    map_view_num = 0
                                else:
                                    try:
                                        point_view_mask = point_mask[..., vid] > 0  # [N, n_points]
                                        map_mask = torch.max(point_view_mask, dim=1)[0]  # [N, ]
                                        map_view_num = int(map_mask.sum())
                                    except IndexError:
                                        print(f"IndexError: point_mask.shape: {point_mask.shape}, vid: {vid}, num_views: {num_views}, orig_num_views: {orig_num_views}")
                                        map_view_num = 0
                                if map_view_num > 0:
                                    view_map_points = frame_map_points[map_mask]
                                    view_map_labels = frame_map_labels[map_mask]
                                    view_points_mask = point_view_mask[map_mask]
                                    

                                    if args.bbox_img_coord:
                                        view_points_img = point_coord_img[map_mask, :, vid]
                                        view_map_points = torch.cat([view_map_points, view_points_img], dim=-1)
                                    
                                    # view_corners = torch.cat([view_corners]*5, dim=0)
                                    # view_labels = torch.cat([view_labels]*5, dim=0)
                                    # box_view_num = box_view_num * 5
    
                                    frame_points_list[vid].append(view_map_points)
                                    frame_label_list[vid].append(view_map_labels)
                                    frame_map_num_list[vid].append(map_view_num)
                                    frame_points_mask_list[vid].append(view_points_mask)
                                    
                                    if args.condition_rope:
                                        view_map_center_img = point_center_img[map_mask, vid, vid]
                                        frame_map_center_list[vid].append(view_map_center_img)                     
                                else:
                                    if args.bbox_img_coord:
                                        frame_points_list[vid].append(torch.zeros((0, args.map_sample_points_num, 6)))
                                    else:
                                        frame_points_list[vid].append(torch.zeros((0, args.map_sample_points_num, 3)))
                                    frame_label_list[vid].append(torch.zeros(0))
                                    frame_map_num_list[vid].append(0)
                                    frame_points_mask_list[vid].append(torch.zeros((0, args.map_sample_points_num)))
                                    if args.condition_rope:
                                        frame_map_center_list[vid].append(torch.zeros((0, 3)))
                        if args.condition_rope:
                            map_sequence.append([frame_points_list, frame_label_list, frame_map_num_list, frame_points_mask_list, frame_map_center_list])
                        else:
                            map_sequence.append([frame_points_list, frame_label_list, frame_map_num_list, frame_points_mask_list])
                else:
                    map_sequence = None
                
                # Prepare GT actions
                action_cluster_ids = None
                gt_future_trajectory = None

                for camera_meta_datas in camera_meta_datas_list:
                    for key in camera_meta_datas:
                        if key not in ['size']:
                            camera_meta_datas[key] = camera_meta_datas[key].to(args.device, non_blocking=True)                         
    
                assert len(camera_meta_datas_list) == inp.shape[2] == num_frames

                # # only for debuging, check if the image sequences are loaded correctly
                # B, V, T, _, H, W = inp.shape
                # BVT3HW = inp.transpose(1,2)
                # BVT3HW[:, :, 3], BVT3HW[:, :, 5] =  BVT3HW[:, :, 5], BVT3HW[:, :, 3]
                # BVT3HW = BVT3HW.reshape(B, -1, 3, H, W)
                # print(BVT3HW.shape)
                # import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
                # import torchvision
                # import numpy as np
                # mean=torch.tensor([0.5, 0.5, 0.5]).view(1,1,3,1,1)
                # std=torch.tensor([0.5, 0.5, 0.5]).view(1,1,3,1,1)
                # BVT3HW = BVT3HW * std + mean
                # vthw = torchvision.utils.make_grid(BVT3HW[0], nrow=6, padding=0, pad_value=1.0)
                # vthw = vthw.clone().permute(1, 2, 0).mul_(255).numpy()
                # vthw = PImage.fromarray(vthw.astype(np.uint8))
                # vthw.save("debug/input_image_temporal_%d.png"%it)
                # if it > 10:
                #     exit(0)

                # Prepare text descriptions
                all_text_features = []
                all_masks = []
                for frame_id in range(len(caption_list)):
                    with torch.no_grad():
                        captions = get_scene_description(caption_list[frame_id])
                        if np.random.rand() < 0.25:
                            captions = [''] * len(captions)
                        tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                        input_ids = tokens.input_ids.cuda(non_blocking=True)
                        mask = tokens.attention_mask.cuda(non_blocking=True)
                        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                        
                        all_text_features.append(text_features)
                        all_masks.append(mask)
                                    
                mask = torch.stack(all_masks, dim=1)
                mask = mask.flatten(0, 1)
                text_features = torch.stack(all_text_features, dim=1)
                text_features = text_features.flatten(0, 1)
                
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)

                kv_compact = []
                for len_i, feat_i in zip(lens, text_features.unbind(0)):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)

                inp = inp.to(args.device, non_blocking=True)
                if it > start_it + 10:
                    telling_dont_kill.early_stop()
                
                # [logging]
                args.cur_it = f'{it+1}/{iters_train}'
                args.last_wei_g = me.meters['tnm'].median
                if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                    _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                    args.dump_log()
                    last_touch = time.time()
                
                # [schedule learning rate]
                wp_it = args.wp * iters_train
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
                
                # [get scheduled hyperparameters]
                progress = g_it / (max_it - 1)
                clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
                
                stepping = (g_it + 1) % args.ac == 0
                step_cnt += int(stepping)
            
            with maybe_record_function('in_training'):
                if args.recurrent_training:
                    grad_norm_t, scale_log2_t = trainer.recurrent_train_step(
                        ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                        metric_lg=me,
                        logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone),
                        inp_BVT3HW=inp,
                        text_cond_tuple=text_cond_tuple,
                        camera_meta_datas_list=camera_meta_datas_list,
                        bbox_sequence=bbox_sequence,
                        action_cluster_ids=action_cluster_ids,
                        gt_future_trajectory=gt_future_trajectory,
                        args=args,
                        remove_casuality=remove_casuality,
                        map_sequence=map_sequence,
                    )
                else:
                    grad_norm_t, scale_log2_t = trainer.train_step(
                        ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                        metric_lg=me,
                        logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone),
                        inp_BVT3HW=inp,
                        text_cond_tuple=text_cond_tuple,
                        camera_meta_datas_list=camera_meta_datas_list,
                        bbox_sequence=bbox_sequence,
                        action_cluster_ids=action_cluster_ids,
                        gt_future_trajectory=gt_future_trajectory,
                        args=args,
                        remove_casuality=remove_casuality,
                        map_sequence=map_sequence,
                    )
                
            del mv_imgs, tokens, text_features, kv_compact
            with maybe_record_function('after_train'):
                me.update(tlr=max_tlr)
    # ============================================= iteration loop ends =============================================
        wandb_utils.log_image("input sample", inp[0], g_it)
        # cleaning garbage every epoch
        gc.collect(), torch.cuda.empty_cache()
        # save local model every epoch
        if (ep + 1) == args.ep or (ep + 1) % args.save_model_ep_freq == 0:
            with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
                local_ckpt_path = saver.sav(args=args, g_it=(g_it+1), next_ep=ep, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15), g_it  # +15: other cost


def main(config=None, experiment_tracker=None):     # # 'pt_le_ft' in train_vae.py is the same as 'pt_le_ft' in train_gpt.py
    
    main_train(config=config, experiment_tracker=experiment_tracker)
    

    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    
    time.sleep(120)


if __name__ == '__main__':
    try:
        main()
    # catch KeyboardInterrupt first.
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Cleaning up...", flush=True)
        
        # If you're using GPUs, you can clean up CUDA memory (optional):
        # torch.cuda.empty_cache()
        
        # Finalize (destroy) the process group
        dist.finalize()
        
        # If you are logging to file, close streams:
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close()
            sys.stderr.close()
        
        # Exit immediately so we do not linger
        sys.exit(0)
    except Exception as _e:
        time.sleep(dist.get_rank() * 1 + random.random() * 0.5)
        try:
            # noinspection PyArgumentList
            print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True)
        except:
            try: print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True)
            except: pass
        if dist.is_master():
            print(f'[err]:\n{_e}')
            traceback.print_exc()
        raise _e
    finally:
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
