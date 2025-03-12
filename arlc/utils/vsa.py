# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import namedtuple
from nvsa.reasoning.vsa_block_utils import (
    pmf2vec,
    binding_circular,
    block_discrete_codebook,
)
from arlc.utils.const import (
    DIM_POSITION_2x2,
    DIM_POSITION_3x3,
    DIM_NUMBER_2x2,
    DIM_NUMBER_3x3,
    DIM_ONEHOT,
)
import torch.nn as nn
from nvsa.reasoning.vsa_block_utils import (
    block_discrete_codebook,
    block_continuous_codebook,
)


def generate_nvsa_codebooks(args, rng):
    """
    Generate the codebooks for NVSA frontend and backend.
    The codebook can also be loaded if it is stored under args.resume/
    """
    backend_cb_cont, _ = block_continuous_codebook(
        device=args.device,
        scene_dim=1024,
        d=args.nvsa_backend_d,
        k=args.nvsa_backend_k,
        rng=rng,
        fully_orthogonal=False,
    )
    backend_cb_discrete, _ = block_discrete_codebook(
        device=args.device, d=args.nvsa_backend_d, k=args.nvsa_backend_k, rng=rng
    )
    return backend_cb_cont, backend_cb_discrete


class VSAConverter(nn.Module):
    def __init__(
        self,
        device,
        constellation,
        dictionary,
        dictionary_type="Discrete",
        context_dim=8,
        attributes_superposition=False,
    ):
        super(VSAConverter, self).__init__()
        self.device = device
        self.constellation = constellation
        self.d = dictionary.shape[1] * dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.dictionary = dictionary
        self.dictionary_type = dictionary_type
        self.compute_attribute_dicts()
        self.context_dim = context_dim
        self.attributes_superposition = attributes_superposition
        if self.attributes_superposition:
            attribute_keys, _ = block_discrete_codebook(
                device=device, d=self.d, k=self.k, scene_dim=5
            )
            self.attribute_keys = nn.Parameter(attribute_keys)

    def compute_attribute_dicts(self):
        if "distribute" in self.constellation or "in_out_four" == self.constellation:
            if "four" in self.constellation:
                DIM_POSITION = DIM_POSITION_2x2
                DIM_NUMBER = DIM_NUMBER_2x2
            else:
                DIM_POSITION = DIM_POSITION_3x3
                DIM_NUMBER = DIM_NUMBER_3x3
            self.position_dictionary = self.dictionary[:DIM_POSITION]
            self.number_dictionary = self.dictionary[:DIM_NUMBER]

    def compute_values(self, scene_prob):
        vsas = {}
        for attr in scene_prob._fields:
            if attr == "position" and (
                "distribute" in self.constellation
                or "in_out_four" == self.constellation
            ):
                vsas[attr] = pmf2vec(self.position_dictionary, scene_prob.position)
            elif attr == "number" and (
                "distribute" in self.constellation
                or "in_out_four" == self.constellation
            ):
                vsas[attr] = pmf2vec(self.number_dictionary, scene_prob.number)
            elif attr in ["position", "number"]:
                vsas[attr] = None
            else:
                vsas[attr] = pmf2vec(
                    self.dictionary[: DIM_ONEHOT + 1], getattr(scene_prob, attr)
                )
        return type(scene_prob)(**vsas)

    def forward(self, scene_prob):
        return self.compute_values(scene_prob)
