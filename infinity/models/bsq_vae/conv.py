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

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, cnn_type="2d", causal_offset=0, temporal_down=False):
        super().__init__()
        self.cnn_type = cnn_type
        self.slice_seq_len = 17
        
        if cnn_type == "2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if cnn_type == "3d":
            if temporal_down == False:
                stride = (1, stride, stride)
            else:
                stride = (stride, stride, stride)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size, kernel_size)
            self.padding = (
                kernel_size[0] - 1 + causal_offset,  # Temporal causal padding
                padding,  # Height padding
                padding  # Width padding
            )
        self.causal_offset = causal_offset
        self.stride = stride
        self.kernel_size = kernel_size
        
    def forward(self, x):
        if self.cnn_type == "2d":
            if x.ndim == 5:
                B, C, T, H, W = x.shape
                x = rearrange(x, "B C T H W -> (B T) C H W")
                x = self.conv(x)
                x = rearrange(x, "(B T) C H W -> B C T H W", T=T)
                return x
            else:
                return self.conv(x)
        if self.cnn_type == "3d":
            assert self.stride[0] == 1 or self.stride[0] == 2, f"only temporal stride = 1 or 2 are supported"
            xs = []
            for i in range(0, x.shape[2], self.slice_seq_len+self.stride[0]-1):
                st = i
                en = min(i+self.slice_seq_len, x.shape[2])
                _x = x[:,:,st:en,:,:]
                if i == 0:
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            self.padding[0], 0))                # Temporal
                else:
                    padding_0 = self.kernel_size[0] - 1
                    _x = F.pad(_x, (self.padding[2], self.padding[2],  # Width
                            self.padding[1], self.padding[1],   # Height
                            padding_0, 0))                      # Temporal
                    _x[:,:,:padding_0,
                        self.padding[1]:_x.shape[-2]-self.padding[1],
                        self.padding[2]:_x.shape[-1]-self.padding[2]] += x[:,:,i-padding_0:i,:,:]
                _x = self.conv(_x)
                xs.append(_x)
            try:
                x = torch.cat(xs, dim=2)
            except:
                device = x.device
                del x
                xs = [_x.cpu().pin_memory() for _x in xs]
                torch.cuda.empty_cache()
                x = torch.cat([_x.cpu() for _x in xs], dim=2).to(device=device)
            return x