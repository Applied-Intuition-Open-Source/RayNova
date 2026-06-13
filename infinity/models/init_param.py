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

import torch.nn as nn


def init_weights(model: nn.Module, conv_std_or_gain: float = 0.02, other_std: float = 0.02):
    """
    :param model: the model to be inited
    :param conv_std_or_gain: how to init every conv layer `m`
        > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
        < 0: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
    :param other_std: how to init every linear layer or embedding layer
        use nn.init.trunc_normal_(m.weight.data, std=other_std)
    """
    skip = abs(conv_std_or_gain) > 10
    if skip: return
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=other_std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain) if conv_std_or_gain > 0 else nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)   # todo: StyleSwin: (..., gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.)
