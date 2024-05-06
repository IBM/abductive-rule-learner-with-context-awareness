# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

from collections import namedtuple
import torch as t
import torch.nn as nn
from arlc.utils.const import (
    DIM_POSITION_2x2,
    DIM_POSITION_3x3,
    DIM_NUMBER_2x2,
    DIM_NUMBER_3x3,
    DIM_TYPE,
    DIM_SIZE,
    DIM_COLOR,
)
from nvsa.reasoning.vsa_block_utils import block_discrete_codebook
from arlc.rule_templates import (
    MLP,
    GeneralLearnableFormula,
    LearnableFormula,
    ExtendedGeneralLearnableFormula,
)
from arlc.utils.vsa import VSAConverter

Scene = namedtuple("Scene", ["position", "number", "type", "size", "color"])


class RuleLevelReasoner(nn.Module):
    def __init__(
        self,
        device,
        constellation,
        model,
        hidden_layers,
        dictionary,
        vsa_conversion=False,
        vsa_selection=False,
        context_superposition=False,
        num_rules=5,
        shared_rules=False,
        program=False,
        rule_type="arlc",
        num_terms=12,
    ):
        super(RuleLevelReasoner, self).__init__()
        self.device = device
        self.constellation = constellation
        self.model = model
        self.program = program
        self.rule_type = rule_type
        self.num_terms = num_terms
        self.d = dictionary.shape[1] * dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.vsa_conversion = vsa_conversion
        self.vsa_selection = vsa_selection
        self.context_superposition = context_superposition
        self.vsa_converter = VSAConverter(
            device, self.constellation, dictionary, dictionary_type="Continuous"
        )
        self.num_rules = num_rules
        self.shared_rules = shared_rules
        self.compute_attribute_rules_sets(hidden_layers)

    def compute_attribute_rules_sets(self, hidden_layers):
        self.type_rules_set = self.size_rules_set = self.color_rules_set = RulesSet(
            self.model,
            hidden_layers,
            self.num_rules,
            2 * self.d,
            -1,
            self.d,
            self.k,
            program=self.program,
            rule_type=self.rule_type,
            num_terms=self.num_terms,
        )
        self.pos_rules_set = self.num_rules_set = self.type_rules_set

    def forward(self, scene_prob, targets=None, distribute=False):
        scene_vsa = self.vsa_converter(scene_prob)
        scene = Scene(
            t.flatten(scene_vsa.position, start_dim=len(scene_vsa.position.shape) - 2)
            if distribute
            else None,
            t.flatten(scene_vsa.number, start_dim=len(scene_vsa.number.shape) - 2)
            if distribute
            else None,
            t.flatten(scene_vsa.type, start_dim=len(scene_vsa.type.shape) - 2),
            t.flatten(scene_vsa.size, start_dim=len(scene_vsa.size.shape) - 2),
            t.flatten(scene_vsa.color, start_dim=len(scene_vsa.color.shape) - 2),
        )
        test_indeces = [2, 5]
        pos_output = self.pos_rules_set(scene.position) if distribute else None
        num_output = self.num_rules_set(scene.number) if distribute else None
        type_output = self.type_rules_set(scene.type)
        size_output = self.size_rules_set(scene.size)
        color_output = self.color_rules_set(scene.color)

        outputs = Scene(pos_output, num_output, type_output, size_output, color_output)

        tests = Scene(
            scene.position[:, test_indeces] if distribute else None,
            scene.number[:, test_indeces] if distribute else None,
            scene.type[:, test_indeces],
            scene.size[:, test_indeces],
            scene.color[:, test_indeces],
        )
        candidates = Scene(
            scene.position[:, 8:] if distribute else None,
            scene.number[:, 8:] if distribute else None,
            scene.type[:, 8:],
            scene.size[:, 8:],
            scene.color[:, 8:],
        )

        return outputs, candidates, tests


class RulesSet(nn.Module):
    def __init__(
        self,
        model,
        hidden_layers,
        num_rules,
        d_in,
        d_out,
        d_vsa,
        k,
        context_superpostion=False,
        context_keys=None,
        program=None,
        rule_type="arlc",
        num_terms=12,
    ):
        super(RulesSet, self).__init__()
        if rule_type == "arlc":
            rule_class = GeneralRule
        else:
            rule_class = Rule
        if program:
            rules = ["constant", "add", "sub", "dist3"] + ["bonus"] * (num_rules - 4)
        else:
            rules = [None] * num_rules
        self.rules = nn.ModuleList(
            [
                rule_class(
                    model,
                    hidden_layers,
                    d_in,
                    d_out,
                    d_vsa,
                    k,
                    context_superpostion,
                    context_keys,
                    program_rule,
                    num_terms=num_terms,
                )
                for program_rule in rules
            ]
        )

    def forward(self, attribute):
        output_list = [
            rule(attribute).reshape((attribute.shape[0], 3, -1)) for rule in self.rules
        ]
        outputs = t.stack(output_list, dim=1)
        return outputs


class Rule(nn.Module):
    def __init__(
        self,
        model,
        hidden_layers,
        d_in,
        d_out,
        d_vsa,
        k,
        context_superposition=False,
        context_keys=None,
        num_terms=12,
    ):
        super(Rule, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d = d_vsa
        self.k = k
        self.context_superposition = context_superposition
        self.context_keys = context_keys
        self.a3_context_indeces = [0, 1, 3, 4, 5, 6, 7]
        self.a6_context_indeces = [0, 1, 2, 3, 4, 6, 7]
        self.a9_context_indeces = [0, 1, 2, 3, 4, 5, 6, 7]
        self.compute_rule(model, hidden_layers)

    def compute_rule(self, model, hidden_layers):
        if model == "MLP":
            self.rule_a3 = MLP(
                self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True
            )
            self.rule_a6 = MLP(
                self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True
            )
            self.rule_a9 = MLP(
                self.d_in, self.d_out, self.d, hidden_layers=hidden_layers, softmax=True
            )
        elif model == "LearnableFormula":
            self.rule_a3 = LearnableFormula(self.d, self.k, self.a3_context_indeces)
            self.rule_a6 = LearnableFormula(self.d, self.k, self.a6_context_indeces)
            self.rule_a9 = LearnableFormula(self.d, self.k, self.a9_context_indeces)

    def forward(self, x):
        a3 = self.rule_a3(x[:, self.a3_context_indeces], self.a3_context_indeces)
        a6 = self.rule_a6(x[:, self.a6_context_indeces], self.a6_context_indeces)
        a9 = self.rule_a9(x[:, self.a9_context_indeces], self.a9_context_indeces)
        return t.cat((a3, a6, a9), dim=1)


class GeneralRule(nn.Module):
    def __init__(
        self,
        model,
        hidden_layers,
        d_in,
        d_out,
        d_vsa,
        k,
        context_superposition=False,
        context_keys=None,
        program_rule=None,
        num_terms=12,
    ):
        super(GeneralRule, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d = d_vsa
        self.k = k
        self.context_superposition = context_superposition
        self.context_keys = context_keys
        self.a3_indeces = [0, 1]
        self.a6_indeces = [3, 4]
        self.a9_indeces = [6, 7]
        self.a3_context_indeces = [[3, 4, 5], [6, 7]]
        self.a6_context_indeces = [[0, 1, 2], [6, 7]]
        self.a9_context_indeces = [[0, 1, 2], [3, 4]]
        if num_terms == 12:
            self.rule = ExtendedGeneralLearnableFormula(
                examples_len=2, context_len=5, k=self.k, program_rule=program_rule
            )
        else:
            self.rule = GeneralLearnableFormula(examples_len=2, context_len=5, k=self.k)

    def forward(self, x):
        a3 = self.rule(
            x=x[:, self.a3_indeces],
            ctx=t.cat([x[:, idx] for idx in self.a3_context_indeces], dim=1),
        )
        a6 = self.rule(
            x=x[:, self.a6_indeces],
            ctx=t.cat([x[:, idx] for idx in self.a6_context_indeces], dim=1),
        )
        a9 = self.rule(
            x=x[:, self.a9_indeces],
            ctx=t.cat([x[:, idx] for idx in self.a9_context_indeces], dim=1),
        )
        return t.cat((a3, a6, a9), dim=1)
