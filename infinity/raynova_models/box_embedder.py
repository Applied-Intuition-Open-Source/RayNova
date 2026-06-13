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

XYZ_MIN = [-100, -100, -10]
XYZ_RANGE = [200, 200, 20]

XYZ6D_MIN = [-100, -100, -10, 0, 0, 0]
XYZ6D_RANGE = [200, 200, 20, 1, 1, 1]

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


class ContinuousBBoxWithTextEmbedding(nn.Module):
    """
    Use continuous bbox coordinate and text embedding with text encoder
    """

    def __init__(
        self,
        n_classes,
        class_token_dim=768,
        output_dim=768,
        trainable_class_token=False,
        embedder_num_freq=0,
        proj_dims=[768, 512, 512],
        mode='xyz-lwh-yaw',
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
        # if self.mode == 'cxyz':
        #     input_dims = 3
        #     output_num = 4  # 4 points
        # elif self.mode == 'all-xyz':
        #     input_dims = 3
        #     output_num = 8  # 8 points
        # elif self.mode == 'owhr':
        #     raise NotImplementedError("Not sure how to do this.")
        # else:
        #     raise NotImplementedError(f"Wrong mode {mode}")
        if self.mode == 'xyz-lwh-yaw':
            input_dims = 7
            output_num = 1  # 4 points
        elif self.mode == 'all-xyz':
            input_dims = 3
            output_num = 8
        elif self.mode == 'all-xyz-6d':
            input_dims = 6
            output_num = 8
        self.minmax_normalize = minmax_normalize
        self.use_text_encoder_init = use_text_encoder_init

        if embedder_num_freq > 0:
            self.fourier_embedder = get_embedder(input_dims, embedder_num_freq, include_input=True)
            logging.info(
                f"[ContinuousBBoxWithTextEmbedding] bbox embedder has "
                f"{self.fourier_embedder.out_dim} dims.")

            self.bbox_proj = nn.Linear(self.fourier_embedder.out_dim * output_num, proj_dims[0])
        else:
            self.fourier_embedder = None
            self.bbox_proj = nn.Sequential(
                nn.Linear(input_dims, proj_dims[0]),
                nn.GELU(approximate='tanh'),
            )
        if len(proj_dims) == 3:
            self.second_linear = nn.Sequential(
                nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
                nn.GELU(approximate='tanh'),
                nn.Linear(proj_dims[1], proj_dims[2]),
                nn.GELU(approximate='tanh'),
                nn.Linear(proj_dims[2], output_dim),
            )
        else:
            self.second_linear = nn.Sequential(
                nn.Linear(proj_dims[0] + class_token_dim, proj_dims[1]),
                nn.GELU(approximate='tanh'),
                nn.Linear(proj_dims[1], output_dim)
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

        # null embedding
        self.null_class_feature = torch.nn.Parameter(
            torch.zeros([class_token_dim]))
        if embedder_num_freq > 0:
            self.null_pos_feature = torch.nn.Parameter(
                torch.zeros([self.fourier_embedder.out_dim * output_num]))
        else:
            self.null_pos_feature = torch.nn.Parameter(
                torch.zeros([input_dims]))

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

    def add_n_uncond_tokens(self, hidden_states, token_num):
        uncond_token = self.forward_feature(
            self.null_pos_feature[None], self.null_class_feature[None])
        uncond_token = uncond_token.repeat(token_num, 1)
        hidden_states = torch.cat([hidden_states, uncond_token], dim=0)
        return hidden_states

    def forward_feature(self, pos_emb, cls_emb):
        emb = self.bbox_proj(pos_emb)

        # combine
        emb = torch.cat([emb, cls_emb], dim=-1)
        emb = self.second_linear(emb)
        return emb

    def forward(self, bboxes: torch.Tensor, classes: torch.LongTensor,
                masks=None, **kwargs):
        """Please do filter before input is needed.

        Args:
            bboxes (torch.Tensor): Expect (B, 8, 3) 
            classes (torch.LongTensor): (B, )

        Return:
            size N x emb_dim=768
        """
        N = classes.shape[0]

        if masks is None:
            masks = torch.ones(N)
        else:
            masks = masks.flatten()
        masks = masks.unsqueeze(-1).type_as(self.null_pos_feature)

        # box
        if self.minmax_normalize:
            bboxes = normalizer(self.mode, bboxes)
        if self.fourier_embedder is not None:
            pos_emb = self.fourier_embedder(bboxes)
            pos_emb = pos_emb.reshape(pos_emb.shape[0], -1)
        else:
            pos_emb = bboxes
        pos_emb = pos_emb.type_as(self.null_pos_feature)
        pos_emb = pos_emb * masks + self.null_pos_feature[None] * (1 - masks)

        # class
        cls_emb = torch.stack([self.class_tokens[i] for i in classes])  # [N, C]
        cls_emb = cls_emb * masks + self.null_class_feature[None] * (1 - masks)

        # combine
        emb = self.forward_feature(pos_emb, cls_emb)

        return emb