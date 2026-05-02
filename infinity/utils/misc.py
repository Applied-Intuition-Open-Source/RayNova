import datetime
import functools
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist
import torch.nn.functional as F

import infinity.utils.dist as dist

os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('America/Los_Angeles')).strftime(fmt)


class DistLogger(object):
    def __init__(self, lg):
        self._lg = lg
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._lg is not None else DistLogger.do_nothing

class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try: import tensorflow_io as tfio
        except: pass
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0
    
    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def loggable(self):
        return self.step == 0 or (self.step + 1) % 500 == 0
    
    def update(self, head='scalar', step=None, **kwargs):
        if step is None:
            step = self.step
            if not self.loggable(): return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            self.writer.add_scalar(f'{head}/{k}', v, step)
    
    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        try:
            self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
        except Exception as e:
            print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')
    
    def log_image(self, tag, img_chw, step=None):
        if step is None:
            step = self.step
            if not self.loggable(): return
        self.writer.add_image(tag, img_chw, step, dataformats='CHW')
    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()


class Low_GPU_usage(object):
    def __init__(self, files, sleep_secs, verbose):
        pass

    def early_stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class TouchingDaemonDontForgetToStartMe(threading.Thread):
    def __init__(self, files: List[str], sleep_secs: int, verbose=False):
        super().__init__(daemon=True)
        self.files = tuple(files)
        self.sleep_secs = sleep_secs
        self.is_finished = False
        self.verbose = verbose
        
        f_back = sys._getframe().f_back
        file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
        self.print_prefix = f' ({file_desc}, line{f_back.f_lineno:-4d}) @daemon@ '
    
    def finishing(self):
        self.is_finished = True
    
    def run(self) -> None:
        kw = {}
        if tdist.is_initialized(): kw['clean'] = True
        
        stt = time.time()
        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] start touching {self.files} per {self.sleep_secs}s ...', **kw)
        while not self.is_finished:
            for f in self.files:
                if os.path.exists(f):
                    try:
                        os.utime(f)
                        fp = open(f, 'a')
                        fp.close()
                    except: pass
            time.sleep(self.sleep_secs)
        
        if self.verbose: print(f'{time_str()}{self.print_prefix}[TouchingDaemon tid={threading.get_native_id()}] finish touching after {time.time()-stt:.1f} secs {self.files} per {self.sleep_secs}s. ', **kw)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0
    
    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)
    
    @property
    def global_avg(self):
        return self.total / (self.count or 1)
    
    @property
    def max(self):
        return max(self.deque) if len(self.deque) else 0
    
    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0
    
    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
    
    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)
        self.iter_end_t = time.time()
        self.log_iters = set()
        self.log_every_iter = False
    
    def update(self, **kwargs):
        # if it != 0 and it not in self.log_iters: return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            # assert isinstance(v, (float, int)), type(v)
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return '  '.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, start_it, max_iters, itrt, log_freq, log_every_iter=False, header=''):    # also solve logging & skipping iterations before start_it
        start_it = start_it % max_iters
        self.log_iters = set(range(start_it, max_iters, log_freq))
        self.log_iters.add(start_it)
        self.log_iters.add(max_iters-1)
        self.log_iters.add(max_iters)
        self.log_every_iter = log_every_iter
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{value:.4f}')
        self.data_time = SmoothedValue(fmt='{value:.3f}')
        header_fmt = header + ':  [{0:' + str(len(str(max_iters))) + 'd}/{1}]'
        
        start_time = time.time()
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for it in range(start_it, max_iters):
                obj = next(itrt)
                if it < start_it: continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  eta: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  T: {self.iter_time.value:.3f}s  dataT: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for it, obj in enumerate(itrt):
                if it < start_it:
                    self.iter_end_t = time.time()
                    continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  eta: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  T: {self.iter_time.value:.3f}s  dataT: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        cost = time.time() - start_time
        cost_str = str(datetime.timedelta(seconds=int(cost)))
        print(f'{header}   Cost of this ep:      {cost_str}   ({cost / (max_iters-start_it):.3f} s / it)', flush=True)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def build_2d_sincos_position_embedding(h, w, embed_dim, temperature=10000., sc=0, verbose=True):    # (1, hw**2, embed_dim)
    # DiT: sc=0
    # DETR: sc=2?
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid([grid_w, grid_h], indexing='ij')
    if sc == 0:
        scale = 1
    elif sc == 1:
        scale = math.pi * 2 / w
    else:
        scale = 1 / w
    grid_w = scale * grid_w.reshape(h*w, 1) # scale * [0, 0, 0, 1, 1, 1, 2, 2, 2]
    grid_h = scale * grid_h.reshape(h*w, 1) # scale * [0, 1, 2, 0, 1, 2, 0, 1, 2]
    
    assert embed_dim % 4 == 0, f'Embed dimension ({embed_dim}) must be divisible by 4 for 2D sin-cos position embedding!'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = (-math.log(temperature) * omega).exp()
    # omega == (1/T) ** (arange(pos_dim) / pos_dim), a vector only dependent on C
    out_w = grid_w * omega.view(1, pos_dim) # out_w: scale * [0*ome, 0*ome, 0*ome, 1*ome, 1*ome, 1*ome, 2*ome, 2*ome, 2*ome]
    out_h = grid_h * omega.view(1, pos_dim) # out_h: scale * [0*ome, 1*ome, 2*ome, 0*ome, 1*ome, 2*ome, 0*ome, 1*ome, 2*ome]
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    if verbose: print(f'[build_2d_sincos_position_embedding @ {hw} x {hw}] scale_type={sc}, temperature={temperature:g}, shape={pos_emb.shape}')
    return pos_emb  # (1, hw**2, embed_dim)


if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    cmap_div = sns.color_palette('icefire', as_cmap=True)
    
    scs = [0, 1, 2]
    temps = [20, 50, 100, 1000]
    reso = 3.0
    RR, CC = len(scs), len(temps)
    plt.figure(figsize=(CC * reso, RR * reso))  # figsize=(16, 16)
    for row, sc in enumerate(scs):
        for col, temp in enumerate(temps):
            name = f'sc={sc}, T={temp}'
            hw, C = 16, 512
            N = hw*hw
            pe = build_2d_sincos_position_embedding(hw, C, temperature=temp, sc=sc, verbose=False)[0] # N, C = 64, 16
            
            hw2 = 16
            N2 = hw2*hw2
            pe2 = build_2d_sincos_position_embedding(hw2, C, temperature=temp, sc=sc, verbose=False)[0] # N, C = 64, 16
            # pe2 = pe2.flip(dims=(0,))
            bchw, bchw2 = F.normalize(pe.view(hw, hw, C).permute(2, 0, 1).unsqueeze(0), dim=1), F.normalize(pe2.view(hw2, hw2, C).permute(2, 0, 1).unsqueeze(0), dim=1)
            dis = [
                f'{F.mse_loss(bchw, F.interpolate(bchw2, size=bchw.shape[-2], mode=inter)).item():.3f}'
                for inter in ('bilinear', 'bicubic', 'nearest')
            ]
            dis += [
                f'{F.mse_loss(F.interpolate(bchw, size=bchw2.shape[-2], mode=inter), bchw2).item():.3f}'
                for inter in ('area', 'nearest')
            ]
            print(f'[{name:^20s}] dis: {dis}')
            """
            [     sc=0, T=20     ] dis: ['0.010', '0.011', '0.011', '0.009', '0.010']
            [    sc=0, T=100     ] dis: ['0.007', '0.007', '0.007', '0.006', '0.007']
            [    sc=0, T=1000    ] dis: ['0.005', '0.005', '0.005', '0.004', '0.005']
            [   sc=0, T=10000    ] dis: ['0.004', '0.004', '0.004', '0.003', '0.004']
            [     sc=1, T=20     ] dis: ['0.007', '0.008', '0.008', '0.007', '0.008']
            [    sc=1, T=100     ] dis: ['0.005', '0.005', '0.005', '0.005', '0.005']
            [    sc=1, T=1000    ] dis: ['0.003', '0.003', '0.003', '0.003', '0.003']
            [   sc=1, T=10000    ] dis: ['0.003', '0.003', '0.003', '0.003', '0.003']
            [     sc=2, T=20     ] dis: ['0.000', '0.000', '0.000', '0.000', '0.000']
            [    sc=2, T=100     ] dis: ['0.000', '0.000', '0.000', '0.000', '0.000']
            [    sc=2, T=1000    ] dis: ['0.000', '0.000', '0.000', '0.000', '0.000']
            [   sc=2, T=10000    ] dis: ['0.000', '0.000', '0.000', '0.000', '0.000']
            Process finished with exit code 0
            """
            
            pe = torch.from_numpy(cmap_div(pe.T.numpy())[:, :, :3])      # C, N, 3
            tar_h, tar_w = 1024, 1024
            pe = pe.repeat_interleave(tar_w//pe.shape[0], dim=0).repeat_interleave(tar_h//pe.shape[1], dim=1)
            plt.subplot(RR, CC, 1+row*CC+col)
            plt.title(name)
            plt.xlabel('hxw'), plt.ylabel('C')
            plt.xticks([]), plt.yticks([])
            plt.imshow(pe.mul(255).round().clamp(0, 255).byte().numpy())
    plt.tight_layout(h_pad=0.02)
    plt.show()


def check_randomness(args):
    U = 16384
    t = torch.zeros(dist.get_world_size(), 4, dtype=torch.float32, device=args.device)
    t0 = torch.zeros(1, dtype=torch.float32, device=args.device).random_(U)
    t[dist.get_rank(), 0] = float(random.randrange(U))
    t[dist.get_rank(), 1] = float(np.random.randint(U))
    t[dist.get_rank(), 2] = float(torch.randint(0, U, (1,))[0])
    t[dist.get_rank(), 3] = float(t0[0])
    dist.allreduce(t)
    for rk in range(1, dist.get_world_size()):
        assert torch.allclose(t[rk - 1], t[rk]), f't={t}'
    del t0, t, U

def get_scene_description(meta_datas):
    template = "This is a driving scene at %s (time %s). %s"
    outputs = []
    for meta_data in meta_datas:
        if meta_data['timeofday'] is None and meta_data['location'] is None:
            outputs.append(meta_data['description'])
        else:
            outputs.append(template%(meta_data['location'], meta_data['timeofday'], meta_data['description']))
    return outputs

def project_corners_to_views(corners, coord_transform, meta_data, return_coord_2d=False, pointwise_mask=False):
    corners = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)  # [N, 8, 4]
    corners = corners @ coord_transform.T
    
    V = meta_data['rot'].shape[0]

    ### First, let's figure out global to img.
    ## Let's make them into 4x4s for my sanity
    # Each of the rots, etc are B x V x 3 or B x V x 3 x 3
    cam_to_cam_aug = meta_data['rot'].new_zeros((V, 4, 4))
    cam_to_cam_aug[:, 3, 3] = 1
    cam_to_cam_aug[:, :3, :3] = meta_data['post_rot']
    cam_to_cam_aug[:, :3, 3] = meta_data['post_trans']

    intrins4x4 = meta_data['rot'].new_zeros((V, 4, 4))
    intrins4x4[:, 3, 3] = 1
    intrins4x4[:, :3, :3] = meta_data['intrins']

    cam_to_lidar_aug = meta_data['rot'].new_zeros((V, 4, 4))
    cam_to_lidar_aug[:, 3, 3] = 1
    cam_to_lidar_aug[:, :3, :3] = meta_data['rot']
    cam_to_lidar_aug[:, :3, 3] = meta_data['trans']

    ## Okay, to go from global to augmed cam (XYD)....
    # We can go from global to (X*D, X*D, D), then we need user to divide by D,
    # Then we can apply the img augs. So, we need to store both.
    # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug

    lidar_to_img = cam_to_cam_aug @ intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
    lidar_to_img = lidar_to_img.to(torch.float32)  # [V, 4, 4]
    
    corners_img = torch.matmul(lidar_to_img[None, None], corners[:, :, None, :, None])  # [N, 8, V, 4, 1]
    corners_img = corners_img.squeeze(-1)[..., :-1]  # [N, 8, V, 3]
    corners_img[..., :2] = corners_img[..., :2] / (corners_img[..., 2:3]+1e-4)
    
    width, height = meta_data['size']
    corners_mask = (corners_img[..., 0] > 0) & (corners_img[..., 0] < width) & (corners_img[..., 1] > 0) & (corners_img[...,1] < height) & (corners_img[...,2] > 0)  # [N, 8, V]
    
    if pointwise_mask:
        box_mask = corners_mask
    else:
        box_mask = torch.max(corners_mask, dim=1)[0]  # [N, V]
    
    if return_coord_2d:
        corners_img_x = corners_img[..., 0] / width
        corners_img_y = corners_img[..., 1] / height
        corners_img_z = corners_img[..., 2] / 50

        corners_img_coord = torch.stack([corners_img_x, corners_img_y, corners_img_z], dim=-1)   # [N, 8, V, 3]

        return box_mask, corners_img_coord
    else:
        return box_mask

def stop_gradient(x):
    if isinstance(x, dict):
        return {k: stop_gradient(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [stop_gradient(v) for v in x]
    elif x is None:
        return x
    else:
        return x.detach()

def clip_cache(x, seq_len_per_frame, max_frame):
    if isinstance(x, dict):
        new_dict = {}
        for k, v in x.items():
            if k != 'sa':
                new_dict[k] = clip_cache(v, seq_len_per_frame, max_frame)
            else:
                new_dict[k] = [None, None]
        return new_dict
    elif isinstance(x, list):
        return [clip_cache(v, seq_len_per_frame, max_frame) for v in x]
    elif x is None:
        return x
    else:
        if x.shape[1] > seq_len_per_frame*max_frame:
            return x[:, -seq_len_per_frame*max_frame:]
        else:
            return x


def remove_cache(x, remove_keys=['sa']):
    if isinstance(x, dict):
        new_cache = {}
        for k, v in x.items():
            if k not in remove_keys:
                new_cache[k] = remove_cache(v)
            else:
                new_cache[k] = [None, None]
        return new_cache
    elif isinstance(x, list):
        return [remove_cache(v) for v in x]
    else:
        return x



