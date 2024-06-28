#
# Copyright (c) IBM Corp. 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def loss(self, output, target):
        loss = 1 - F.cosine_similarity(output, target, dim=-1)
        return loss

    def forward(self, output, target):
        loss = self.loss(output, target)
        loss = loss.mean(dim=-1)
        return loss

    def score(self, output, targets):
        losses = self.loss(output, targets)
        score = -losses
        return score


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def loss(self, output, target):
        loss = torch.sum(-target * torch.log(output), dim=-1)
        return loss

    def forward(self, output, target):
        loss = self.loss(output, target)
        loss = loss.mean(dim=-1)
        return loss

    def score(self, output, targets):
        losses = self.loss(output, targets)
        score = -losses
        return score


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def loss(self, output, target):
        output_normalized = output / output.sum(dim=-1, keepdim=True)
        target_normalized = target / target.sum(dim=-1, keepdim=True)
        epsilon = 1e-10
        loss = torch.sum(
            output_normalized
            * (
                torch.log(output_normalized + epsilon)
                - torch.log(target_normalized + epsilon)
            ),
            dim=-1,
        )
        return loss

    def forward(self, output, target):
        loss = self.loss(output, target)
        loss = loss.mean(dim=-1)
        return loss

    def score(self, output, targets):
        losses = self.loss(output, targets)
        scores = -losses
        return scores
