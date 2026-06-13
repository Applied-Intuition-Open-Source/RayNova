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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from infinity.models.embedder import get_embedder

XYZ_MIN = [-50, -50, -5]
XYZ_RANGE = [50, 50, 5]

XYZ6D_MIN = [-50, -50, -5, 0, 0, 0]
XYZ6D_RANGE = [100, 100, 10, 1, 1, 50]

def normalizer(mode, data):
    if mode == 'all-xyz':
        # data in format of (N, 8, 3):
        mins = torch.as_tensor(
            XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'all-xyz-6d':
        # data in format of (N, 8, 6):
        mins = torch.as_tensor(
            XYZ6D_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(
            XYZ6D_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == 'owhr':
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


class MapWithTextEmbedding(nn.Module):
    """
    Use map sampled points and text embedding with text encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        output_dim=768,
        trainable_class_token=False,
        proj_dims=[64, 64, 128, 512, 512],
        mode='all-xyz',
        center_normalize=True,
        minmax_normalize=True,
        use_text_encoder_init=True,
        **kwargs,
    ):
        """
        Args:
            mode (str, optional): cxyz -> all points; all-xyz -> all points;
                owhr -> center, l, w, h, z-orientation.
        """
        super().__init__()

        self.mode = mode
        if self.mode == 'all-xyz':
            input_dims = 3
        elif self.mode == 'all-xyz-6d':
            input_dims = 6
        self.center_normalize = center_normalize
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        self.point_proj = nn.Sequential(
            nn.Linear(input_dims, proj_dims[0]),
            nn.LayerNorm(proj_dims[0]),
            nn.GELU(approximate='tanh'),
            nn.Linear(proj_dims[0], proj_dims[1]),
            nn.LayerNorm(proj_dims[1]),
            nn.GELU(approximate='tanh'),
            nn.Linear(proj_dims[1], proj_dims[2]),
        )

        if self.center_normalize:
            global_input_dims = proj_dims[2]*3
        else:
            global_input_dims = proj_dims[2]*2
        
        if self.center_normalize:
            self.center_proj = nn.Sequential(
                nn.Linear(input_dims, proj_dims[0]),
                nn.LayerNorm(proj_dims[0]),
                nn.GELU(approximate='tanh'),
                nn.Linear(proj_dims[0], proj_dims[1]),
                nn.LayerNorm(proj_dims[1]),
                nn.GELU(approximate='tanh'),
                nn.Linear(proj_dims[1], proj_dims[2]),
            )
        self.second_linear = nn.Sequential(
            nn.Linear(global_input_dims + class_token_dim, proj_dims[3]),
            nn.LayerNorm(proj_dims[3]),
            nn.GELU(approximate='tanh'),
            nn.Linear(proj_dims[3], proj_dims[4]),
            nn.LayerNorm(proj_dims[4]),
            nn.GELU(approximate='tanh'),
            nn.Linear(proj_dims[4], output_dim),
        )

        # for class token
        self._class_tokens_set_or_warned = not self.use_text_encoder_init
        if trainable_class_token:
            # parameter is trainable, buffer is not
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_parameter("_class_tokens", nn.Parameter(class_tokens))
        else:
            class_tokens = torch.randn(n_classes, class_token_dim)
            self.register_buffer("_class_tokens", class_tokens)
            if not self.use_text_encoder_init:
                logging.warn(
                    "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not"
                    " trainable but you set `use_text_encoder_init` to False. "
                    "Please check your config!")

    @property
    def class_tokens(self):
        if not self._class_tokens_set_or_warned:
            logging.warn(
                "[ContinuousBBoxWithTextEmbedding] Your class_tokens is not "
                "trainable and used without initialization. Please check your "
                "training code!")
            self._class_tokens_set_or_warned = True
        return self._class_tokens

    def prepare(self, tokenizer, text_encoder, class_names):
        if self.use_text_encoder_init:
            self.set_category_token(tokenizer, text_encoder, class_names)
        else:
            logging.info("[ContinuousBBoxWithTextEmbedding] Your class_tokens "
                         "initilzed with random.")

    @torch.no_grad()
    def set_category_token(self, tokenizer, text_encoder, class_names):
        logging.info("[ContinuousBBoxWithTextEmbedding] Initialzing your "
                     "class_tokens with text_encoder")
        self._class_tokens_set_or_warned = True
        device = self.class_tokens.device
        for idx, name in enumerate(class_names):
            inputs = tokenizer([name.lower()], padding='do_not_pad', return_tensors='pt')
            inputs = inputs.input_ids.to(device)
            # there are two outputs: last_hidden_state and pooler_output
            # we use the pooled version.
            hidden_state = text_encoder(inputs).last_hidden_state[0] 
            hidden_state = torch.mean(hidden_state, dim=0)
            self.class_tokens[idx].copy_(hidden_state)

    def forward_feature(self, points, cls_emb, masks):
        # masks: (B, N)
        if self.center_normalize:
            center = torch.mean(points, dim=1) # (B, 1, input_dims)
            points = points - center[:, None]
        
        points_emb = self.point_proj(points) # (B, N, C)

        if masks is None:
            masks = torch.ones(points_emb.shape[0], points_emb.shape[1]).type_as(points_emb)
        masks = masks.unsqueeze(-1).repeat(1, 1, points_emb.shape[-1])  # (B, N, C)

        points_emb_mean = torch.sum(points_emb * masks, dim=1) / masks.sum(dim=1) # (B, C)
        points_emb = torch.where(masks>0, points_emb, -torch.inf) # (B, N, C)
        points_emb_max = torch.max(points_emb, dim=1)[0] # (B, C)
        
        points_emb = torch.cat([points_emb_mean, points_emb_max], dim=-1) # (B, C*2)
        
        if self.center_normalize:
            center_emb = self.center_proj(center)  # (B, C)
            points_emb = torch.cat([points_emb, center_emb], dim=-1)

        # combine
        emb = torch.cat([points_emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, points: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            points (torch.Tensor): Expect (B, N, 3) 
            classes (torch.LongTensor): (B, )

        Return:
            size N x emb_dim=768
        """
        N = classes.shape[0]
        
        # box
        if self.minmax_normalize:
            points = normalizer(self.mode, points)
        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes])  # [N, C]

        # combine
        emb = self.forward_feature(points, cls_emb, masks)

        return emb