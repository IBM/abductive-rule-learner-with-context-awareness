# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RuleSelector(nn.Module):
    def __init__(self, loss_fn, temperature, rule_selector="sample"):
        super(RuleSelector, self).__init__()
        self.loss_fn = loss_fn
        self.temperature = temperature
        self.train_mode = True

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def attribute_forward(self, outputs, tests, candidates=None, targets=None):
        if self.train_mode:
            tests = (
                t.cat(
                    (
                        tests,
                        candidates[t.arange(candidates.shape[0]), targets].unsqueeze(1),
                    ),
                    dim=1,
                )
                .unsqueeze(1)
                .expand(-1, outputs.shape[1], -1, -1)
            )
            scores = self.loss_fn.score(outputs, tests).mean(dim=-1)
            weights = F.softmax(scores / self.temperature, dim=-1)
        else:
            tests = tests.unsqueeze(1).expand(-1, outputs.shape[1], -1, -1)
            scores = self.loss_fn.score(outputs[:, :, :2], tests).mean(dim=-1)
            weights = F.softmax(scores / self.temperature, dim=-1)
        outputs = t.einsum("ijkh,ij->ikh", outputs, weights)
        return outputs, weights

    def _entropy(self, dist):
        dist = dist.detach().cpu().numpy()
        entropy = -(dist * np.log(dist))
        return entropy[~np.isnan(entropy)].sum() / dist.shape[0]

    def forward(self, outputs, tests, candidates=None, targets=None, use_position=True):
        res = {}
        rules = {}
        weights = {}
        for attr in outputs._fields:
            if attr in ["position", "number"] and (
                not use_position or outputs.position is None
            ):
                res[attr] = rules[attr] = None
                continue
            res[attr], weights[attr] = self.attribute_forward(
                getattr(outputs, attr),
                getattr(tests, attr),
                getattr(candidates, attr, None),
                targets,
            )
        res = type(outputs)(**res)
        entropy_attr = {k: self._entropy(v) for k, v in weights.items()}
        return res, entropy_attr
