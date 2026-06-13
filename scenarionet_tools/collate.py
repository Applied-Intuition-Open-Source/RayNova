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

from nuscenes_tools.nuscenes_utils.collate import collate as nuscenes_collate

def collate(batch, samples_per_gpu=1):
    output_batch = []
    for tid in range(len(batch[0])):
        sample_batch = []
        for sample_id in range(len(batch)):
            sample_batch.append(batch[sample_id][tid])
        output_batch.append(nuscenes_collate(sample_batch))
    
    return output_batch