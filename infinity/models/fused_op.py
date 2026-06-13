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

import gc
from copy import deepcopy
from typing import Union

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.compile(fullgraph=True)
def fused_rms_norm(x: torch.Tensor, weight: nn.Parameter, eps: float):
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps))) * weight


@torch.compile(fullgraph=True)
def fused_ada_layer_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = F.layer_norm(input=x, normalized_shape=(C,), weight=None, bias=None, eps=eps)
    return x.mul(scale.add(1)).add_(shift)


@torch.compile(fullgraph=True)
def fused_ada_rms_norm(C: int, eps: float, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    x = x.float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(eps)))
    return x.mul(scale.add(1)).add_(shift)
