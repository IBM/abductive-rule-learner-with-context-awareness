# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])

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
            # append the ground truth answer panel to the tests. MICHAEL: what is outputs.shape[1]
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

    def forward(self, outputs, tests, candidates=None, targets=None, use_position=True):
        if outputs.position is not None and use_position:
            pos_output, pos_rules = self.attribute_forward(
                outputs.position,
                tests.position,
                getattr(candidates, "position", None),
                targets,
            )
            num_output, num_rules = self.attribute_forward(
                outputs.number,
                tests.number,
                getattr(candidates, "number", None),
                targets,
            )
        else:
            pos_output, pos_rules = None, None
            num_output, num_rules = None, None
        type_output, type_rules = self.attribute_forward(
            outputs.type, tests.type, getattr(candidates, "type", None), targets
        )
        size_output, size_rules = self.attribute_forward(
            outputs.size, tests.size, getattr(candidates, "size", None), targets
        )
        color_output, color_rules = self.attribute_forward(
            outputs.color, tests.color, getattr(candidates, "color", None), targets
        )
        rules = [pos_rules, num_rules, color_rules, size_rules, type_rules]
        outputs = Scene(pos_output, num_output, type_output, size_output, color_output)
        return outputs, rules
