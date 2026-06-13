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


class ProbabilisticDataLoader:
    def __init__(self, dataloader1, dataloader2, prob1=0.1):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
        self.prob1 = prob1
        self.prob2 = 1.0 - prob1

        self.iter1 = iter(dataloader1)
        self.iter2 = iter(dataloader2)

    def __iter__(self):
        self.iter1 = iter(self.dataloader1)
        self.iter2 = iter(self.dataloader2)
        return self

    def __next__(self):
        p = random.random()
        if p < self.prob1:
            return next(self.iter1)
        else:
            return next(self.iter2)
