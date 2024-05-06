# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import namedtuple
from nvsa.reasoning.vsa_block_utils import pmf2vec, block_discrete_codebook
from arlc.utils.const import DIM_POSITION_2x2, DIM_POSITION_3x3,DIM_NUMBER_2x2,DIM_NUMBER_3x3,DIM_TYPE,DIM_SIZE,DIM_COLOR
import torch.nn as nn
from nvsa.reasoning.vsa_block_utils import block_discrete_codebook, block_continuous_codebook

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])


def generate_nvsa_codebooks(args, rng):
    """
    Generate the codebooks for NVSA frontend and backend.
    The codebook can also be loaded if it is stored under args.resume/
    """
    backend_cb_cont, _ = block_continuous_codebook( device=args.device, scene_dim=511, d=args.nvsa_backend_d, k=args.nvsa_backend_k, rng=rng, fully_orthogonal=False,  )
    backend_cb_discrete, _ = block_discrete_codebook( device=args.device, d=args.nvsa_backend_d, k=args.nvsa_backend_k, rng=rng  )
    return backend_cb_cont, backend_cb_discrete


class VSAConverter(nn.Module):
    def __init__( self, device, constellation, dictionary, dictionary_type="Discrete", context_dim=8, attributes_superposition=False,  ):
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
            attribute_keys, _ = block_discrete_codebook(device=device, d=self.d, k=self.k, scene_dim=5)
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
        if self.type == "Discrete":
            self.type_dictionary = self.dictionary[: DIM_TYPE + 1]
            self.size_dictionary = self.dictionary[: DIM_SIZE + 1]
        else:
            self.type_dictionary = self.dictionary[1 : DIM_TYPE + 2]
            self.size_dictionary = self.dictionary[1 : DIM_SIZE + 2]
        self.color_dictionary = self.dictionary[: DIM_COLOR + 1]

    def compute_values(self, scene_prob):
        if "distribute" in self.constellation or "in_out_four" == self.constellation:
            position = pmf2vec(self.position_dictionary, scene_prob.position_prob)
            number = pmf2vec(self.number_dictionary, scene_prob.number_prob)
        else:
            position = None
            number = None
        type = pmf2vec(self.type_dictionary, scene_prob.type_prob)
        size = pmf2vec(self.size_dictionary, scene_prob.size_prob)
        color = pmf2vec(self.color_dictionary, scene_prob.color_prob)
        return Scene(position, number, type, size, color)

    def forward(self, scene_prob):
        return self.compute_values(scene_prob)
