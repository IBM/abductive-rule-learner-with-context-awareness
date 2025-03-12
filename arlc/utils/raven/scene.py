# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import OrderedDict, namedtuple
from itertools import product
import numpy as np
import torch
import arlc.utils.general as utils


class SceneEngine:
    def __init__(self, number_slots, device):
        self.device = device
        self.num_slots = number_slots
        self.positions = list(product(range(2), repeat=self.num_slots))
        # assume nonempty
        start_index = 1
        position2number = np.sum(self.positions[start_index:], axis=1)
        # note the correspondence of positions: first digit from the left corresponds to part one
        self.positions = torch.tensor(self.positions[start_index:], dtype=torch.int).to(
            self.device
        )
        self.dim_position = self.positions.shape[0]
        self.num_pos_index_map = OrderedDict()
        for i in range(start_index, self.num_slots + 1):
            self.num_pos_index_map[i] = torch.tensor(
                list(
                    filter(
                        lambda idx: position2number[idx] == i,
                        range(len(position2number)),
                    )
                ),
                dtype=torch.long,
            ).to(self.device)

    def compute_scene_prob(self, **attribute_logprobs):
        position_prob, position_logprob = self.compute_position_prob(
            attribute_logprobs.pop("exist")
        )
        number_prob, number_logprob = self.compute_number_prob(position_prob)
        SceneProb = namedtuple(
            "SceneProb", ["position", "number"] + [k for k in attribute_logprobs.keys()]
        )
        SceneLogProb = namedtuple(
            "SceneLogProb",
            ["position", "number"] + [k for k in attribute_logprobs.keys()],
        )
        attr_probs = {
            k: self.compute_attribute_prob(v, position_logprob)
            for k, v in attribute_logprobs.items()
        }
        att_logprobs = {k: utils.log(v) for k, v in attr_probs.items()}
        return (
            SceneProb(position_prob, number_prob, **attr_probs),
            SceneLogProb(position_logprob, number_logprob, **att_logprobs),
        )

    def compute_position_prob(self, exist_logprob):
        batch_size = exist_logprob.shape[0]
        num_panels = exist_logprob.shape[1]
        exist_logprob = exist_logprob.unsqueeze(2).expand(
            -1, -1, self.dim_position, -1, -1
        )
        index = (
            self.positions.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_panels, -1, -1)
            .unsqueeze(-1)
            .type(torch.long)
        )
        position_logprob = torch.gather(
            exist_logprob, -1, index
        )  # (batch_size, num_panels, self.dim_position, slots, 1)
        position_logprob = torch.sum(
            position_logprob.squeeze(-1), dim=-1
        )  # (batch_size, num_panels, self.dim_position)
        position_prob = torch.exp(position_logprob)
        # assume nonempty: all zero state is filtered out
        position_prob = utils.normalize(position_prob)[0]
        position_logprob = utils.log(position_prob)
        return position_prob, position_logprob

    def compute_number_prob(self, position_prob):
        all_num_prob = []
        for _, indices in self.num_pos_index_map.items():
            num_prob = torch.sum(position_prob[:, :, indices], dim=-1, keepdim=True)
            all_num_prob.append(num_prob)
        number_prob = torch.cat(all_num_prob, dim=-1)
        return number_prob, utils.log(number_prob)

    def compute_attribute_prob(self, logprob, position_logprob):
        batch_size = logprob.shape[0]
        num_panels = logprob.shape[1]
        index = (
            self.positions.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_panels, -1, -1)
            .unsqueeze(-1)
            .type(torch.float)
        )
        logprob = logprob.unsqueeze(2).expand(-1, -1, self.dim_position, -1, -1)
        logprob = (
            index * logprob
        )  # (batch_size, num_panels, self.dim_position, slots, DIM_TYPE)
        logprob = torch.sum(logprob, dim=3) + position_logprob.unsqueeze(-1)
        prob = torch.exp(logprob)
        prob = torch.sum(prob, dim=2)
        inconsist_prob = 1.0 - torch.clamp(
            torch.sum(prob, dim=-1, keepdim=True), max=1.0
        )  # clamp for numerical stability
        prob = torch.cat([prob, inconsist_prob], dim=-1)
        return torch.nan_to_num(prob, nan=0.0)
