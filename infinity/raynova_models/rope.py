from functools import partial
from typing import Callable, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


def apply_rope_cond_keys(cond_k, k_center, scale_schedule, base=10000.0, max_size=84):
    _, max_height, max_width = scale_schedule[-1]
    if max_height > max_width:
        height = max_size
        width = max_size / max_height * max_width
    else:
        width = max_size
        height = max_size / max_width * max_height
    device = cond_k.device
    half_dim = cond_k.shape[-1] // 2
    t_width = width * k_center[..., 0]
    t_height = height * k_center[..., 1]

    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim))

    freqs_height = torch.outer(t_height, inv_freq) 
    freqs_width = torch.outer(t_width, inv_freq)
    
    freqs_hw = torch.cat([freqs_height, freqs_width], dim=-1)
    
    freqs_hw = torch.stack([torch.cos(freqs_hw), torch.sin(freqs_hw)], dim=0)
    cond_k = cond_k.reshape(*cond_k.shape[:-1], -1, 2)
    
    freqs_hw = freqs_hw[:, :, None]

    cond_k_rope = torch.stack(
        [
            freqs_hw[0] * cond_k[...,0] - freqs_hw[1] * cond_k[...,1],
            freqs_hw[1] * cond_k[...,0] + freqs_hw[0] * cond_k[...,1],
        ], dim=-1
    )
    cond_k_rope = cond_k_rope.reshape(*cond_k.shape[:-2], -1)
    
    return cond_k_rope


def apply_rope_time(qk_time, seq_data, base=10000.0, scale=1):
    device = qk_time.device
    dim = qk_time.shape[-1]
    seq_data = seq_data * scale
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)) # namely theta
    freqs_time = seq_data[..., None] * inv_freq[None, None, :]
    freqs_time = torch.stack([torch.cos(freqs_time), torch.sin(freqs_time)], dim=0)  # (2, batch_size, seq_len, quat_head_dim//2)
    freqs_time = freqs_time[:, :, None]

    qk_time = qk_time.reshape(*qk_time.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, quat_head_dim//2, 2)
    qk_time = torch.stack([
        freqs_time[0] * qk_time[...,0] - freqs_time[1] * qk_time[...,1],
        freqs_time[1] * qk_time[...,0] + freqs_time[0] * qk_time[...,1],
        ], dim=-1
    ) # (2, batch_size, heads, seq_len, quat_head_dim//2, 2), here stack + reshape should not be concate
    qk_time = qk_time.reshape(*qk_time.shape[:-2], -1) #(2, batch_size, heads, seq_len, quat_head_dim)
    q_time, k_time = qk_time.unbind(dim=0) # (batch_size, heads, seq_len, quat_head_dim)

    return q_time, k_time


def apply_rope_camera(qk_cam, view_meta_data):

    q_cam, k_cam = qk_cam.unbind(dim=0) # (batch_size, heads, seq_len, quat_head_dim)
    q_cam = q_cam.reshape(*q_cam.shape[:-1], -1, 4) #(batch_size, heads, seq_len, quat_head_dim//4, 4)
    k_cam = k_cam.reshape(*k_cam.shape[:-1], -1, 4) #(batch_size, heads, seq_len, quat_head_dim//4, 4)
    
    P = view_meta_data["intrinsic_norm"] @ view_meta_data["extrinsic"] # (batch_size, seq_len, 4, 4)
    P_T = P.transpose(-1, -2).to(q_cam.dtype) # (batch_size, seq_len, 4, 4)
    P_inv = torch.inverse(P.to(torch.float32)).to(q_cam.dtype) # (batch_size, seq_len, 4, 4)

    q_cam = P_T[:, None, :, None] @ q_cam[..., None] # (batch_size, heads, seq_len, quat_head_dim//4, 4, 1)
    k_cam = P_inv[:, None, :, None] @ k_cam[..., None] # (batch_size, heads, seq_len, quat_head_dim//4, 4, 1)
    q_cam = q_cam.squeeze(-1) # (batch_size, heads, seq_len, quat_head_dim//4, 4)
    k_cam = k_cam.squeeze(-1) # (batch_size, heads, seq_len, quat_head_dim//4, 4)

    q_cam = q_cam.reshape(*q_cam.shape[:-2], -1) # (batch_size, heads, seq_len, quat_head_dim)
    k_cam = k_cam.reshape(*k_cam.shape[:-2], -1) # (batch_size, heads, seq_len, quat_head_dim)
    
    return q_cam, k_cam

def apply_rope_xy_precompute(qk_rope, seq_len, rope2d_freqs_grid, scale_ind, scale_schedule, num_views, timesteps):
    start = 0
    dim = qk_rope.shape[-1]
    if scale_ind >= 1:
        assert len(scale_schedule[0]) == 3
        start = np.sum([item[0] * item[1] * item[2] * num_views for item in scale_schedule[:scale_ind]])
    rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk_rope.device)

    # if start != 0 or seq_len != 472 or rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4] != 11252:
    #     print("start", start)
    #     print("seq_len", seq_len)
    #     print("rope2d_freqs_grid[str(tuple(scale_schedule))].shape", rope2d_freqs_grid[str(tuple(scale_schedule))].shape)
    #     import pdb; pdb.set_trace()
    assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4], f"{start}, {seq_len}, {rope2d_freqs_grid[str(tuple(scale_schedule))].shape}, {qk_rope.shape}"
    rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
    # print("rope_cache", rope_cache.shape)
    # print("qk", qk.shape)
    # exit(0)
    rope_cache = rope_cache.split(dim//2, dim=-1)[0]
    rope_cache = torch.cat([rope_cache]*timesteps, dim=-2)
    
    qk_rope = qk_rope.reshape(*qk_rope.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, half_head_dim//2, 2)
    qk_rope = torch.stack([
        rope_cache[0] * qk_rope[...,0] - rope_cache[1] * qk_rope[...,1],
        rope_cache[1] * qk_rope[...,0] + rope_cache[0] * qk_rope[...,1],
    ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim//2, 2), here stack + reshape should not be concate
    qk_rope = qk_rope.reshape(*qk_rope.shape[:-2], -1) #(2, batch_size, heads, seq_len, half_head_dim)
    qk_rope = qk_rope.permute(0, 1, 3, 2, 4).squeeze(0)  # (1, batch_size, seq_len, heads, half_head_dim)
    return qk_rope, rope_cache


def apply_rope_xy(q_rope, seq_len, rope2d_freqs_grid, scale_ind, scale_schedule, num_views, timesteps, base=10000.0, max_size=84):
    assert timesteps == 1, f"timesteps={timesteps} != 1"
    assert num_views == 1, f"num_views={num_views} != 1"
    cur_len = 0
    img_coords = []
    device = q_rope.device
    _, max_height, max_width = scale_schedule[-1]
    if max_height > max_width:
        height = max_size
        width = max_size / max_height * max_width
    else:
        width = max_size
        height = max_size / max_width * max_height
    scale_ind = max(0, scale_ind)
    img_coords = []
    for si in range(scale_ind, len(scale_schedule)):
        _, ph, pw = scale_schedule[si]
        xs = torch.arange(pw) + 0.5
        ys = torch.arange(ph) + 0.5
        xs = xs * width / pw
        ys = ys * height / ph
        xs = xs[None, :].repeat(ph, 1)
        ys = ys[:, None].repeat(1, pw)
        img_coords_scale = torch.stack([ys, xs], dim=-1).flatten(0, 1)  # [ph*pw, 2]
        img_coords_scale = torch.cat([img_coords_scale]*num_views, dim=0)
        img_coords.append(img_coords_scale)
        
        cur_len += ph * pw * num_views
        assert cur_len <= seq_len, f"cur_len={cur_len} > seq_len={seq_len}, si={si}"
        if cur_len == seq_len:
            break
        
    img_coords = torch.cat(img_coords, dim=0)  # [L_sf, 2]
    img_coords = img_coords.to(device)
    
    half_dim = q_rope.shape[-1] // 2
        
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)) # [dim //4]
    freqs_xy = img_coords[..., None] * inv_freq[None, None, :]  # [L_sf, 2, dim // 4]
    freqs_xy = freqs_xy.flatten(-2, -1)  # [L_sf, dim // 2]
    freqs_xy = torch.stack([torch.cos(freqs_xy), torch.sin(freqs_xy)], dim=0)  # (2, L_sf, dim // 2)
    
    q_rope = q_rope.reshape(*q_rope.shape[:-1], -1, 2) # (batch_size, heads, seq_len, dim//2, 2)

    q_rope = torch.stack([
        freqs_xy[0] * q_rope[...,0] - freqs_xy[1] * q_rope[...,1],
        freqs_xy[1] * q_rope[...,0] + freqs_xy[0] * q_rope[...,1],
        ], dim=-1
    ) # (batch_size, heads, seq_len, dim//2, 2), here stack + reshape should not be concate

    q_rope = q_rope.reshape(*q_rope.shape[:-2], -1) #(batch_size, heads, seq_len, dim)
    q_rope = q_rope.permute(0, 2, 1, 3)  # (batch_size, seq_len, heads, dim)
    
    return q_rope, freqs_xy

def apply_rope_condition(latent_q, cond_k, k_center, rope2d_freqs_grid, scale_ind, scale_schedule, seq_len):
    batch_size = latent_q.shape[0] // seq_len
    latent_q = latent_q.reshape(batch_size, seq_len, *latent_q.shape[1:])  # (batch_size, seq_len, heads, quat_head_dim)
    latent_q = latent_q.permute(0, 2, 1, 3)  # (batch_size, heads, seq_len, quat_head_dim)
    latent_q_rope, rope_1 = apply_rope_xy(latent_q, seq_len, rope2d_freqs_grid, scale_ind, scale_schedule, num_views=1, timesteps=1)
    # latent_q_rope_precompute, rope_2 = apply_rope_xy_precompute(latent_q, seq_len, rope2d_freqs_grid, scale_ind, scale_schedule, num_views=1, timesteps=1)

    latent_q_rope = latent_q_rope.flatten(0, 1)
    
    cond_k = apply_rope_cond_keys(cond_k, k_center, scale_schedule)
    return latent_q_rope, cond_k
    
    

def apply_rotary_emb_with_camera(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind, view_meta_data=None, timesteps=1, num_views=1):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    seq_len = qk.shape[3] // timesteps
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    half_head_dim = qk.shape[-1] // 2
    quat_head_dim = half_head_dim // 2
    if view_meta_data['remove_xy']:
        qk_cam, qk_time = qk.split(half_head_dim, dim=-1)
    else:
        qk_rope, qk_half = qk.split(half_head_dim, dim=-1)
        qk_cam, qk_time = qk_half.split(quat_head_dim, dim=-1)
    with torch.autocast(device_type=device_type, enabled=False):
        if not view_meta_data['remove_xy']:
            qk_rope = apply_rope_xy(qk_rope, seq_len, rope2d_freqs_grid, scale_ind, scale_schedule, num_views, timesteps)
            q_rope, k_rope = qk_rope.unbind(dim=0) # (batch_size, heads, seq_len, half_head_dim)
        q_cam, k_cam = apply_rope_camera(qk_cam, view_meta_data)
        q_time, k_time = apply_rope_time(qk_time, view_meta_data["timestep"], scale=10)

        if view_meta_data['remove_xy']:
            q = torch.cat([q_cam, q_time], dim=-1) # (batch_size, heads, seq_len, head_dim)
            k = torch.cat([k_cam, k_time], dim=-1) # (batch_size, heads, seq_len, head_dim)
        else:
            q = torch.cat([q_rope, q_cam, q_time], dim=-1) # (batch_size, heads, seq_len, head_dim)
            k = torch.cat([k_rope, k_cam, k_time], dim=-1) # (batch_size, heads, seq_len, head_dim)

    return q, k


def apply_rotary_emb_with_camera_ray6D(q, k, view_meta_data=None):
    
    freq_base = 100.0
    time_scale = 30
    ray_scale = 50
    camera_center_scale = 2
    
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    oct_head_dim = qk.shape[-1] // 8
    assert oct_head_dim % 2 == 0

    qk_split = qk.split(oct_head_dim, dim=-1)
    
    qk_time = torch.cat([qk_split[0], qk_split[1]], dim=-1)
    q_time, k_time = apply_rope_time(qk_time, view_meta_data["timestep"], scale=time_scale, base=freq_base)
    
    q_ray = []
    k_ray = []
    for i in range(3):
        q_ray_dim, k_ray_dim = apply_rope_time(qk_split[i+2], view_meta_data["ray"][..., i], scale=ray_scale, base=freq_base)
        q_ray.append(q_ray_dim)
        k_ray.append(k_ray_dim)
    q_ray = torch.cat(q_ray, dim=-1)
    k_ray = torch.cat(k_ray, dim=-1)
    
    q_cam_center = []
    k_cam_center = []
    if 'prel' in view_meta_data['view_embed_type']:
        camera_center = torch.cross(view_meta_data["camera_center"], view_meta_data["ray"], dim=-1)
    else:
        camera_center = view_meta_data["camera_center"]
    
    for i in range(3):
        q_cam_center_dim, k_cam_center_dim = apply_rope_time(qk_split[i+5], camera_center[..., i], scale=camera_center_scale, base=freq_base)
        q_cam_center.append(q_cam_center_dim)
        k_cam_center.append(k_cam_center_dim)
    q_cam_center = torch.cat(q_cam_center, dim=-1)
    k_cam_center = torch.cat(k_cam_center, dim=-1)
    
    q = torch.cat([q_time, q_ray, q_cam_center], dim=-1) # (batch_size, heads, seq_len, head_dim)
    k = torch.cat([k_time, k_ray, k_cam_center], dim=-1) # (batch_size, heads, seq_len, head_dim)

    return q, k