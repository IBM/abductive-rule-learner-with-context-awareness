# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import torch as t
import torch.nn as nn
from arlc.rule_templates import (
    ExtendedGeneralLearnableFormula,
    IravenxGeneralLearnableFormula,
    IravenVGeneralLearnableFormula,
)
from arlc.utils.vsa import VSAConverter


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
        n=3,
    ):
        super(RuleLevelReasoner, self).__init__()
        self.device = device
        self.constellation = constellation
        self.model = model
        self.program = program
        self.rule_type = rule_type
        self.num_terms = num_terms
        self.num_panels = n
        self.d = dictionary.shape[1] * dictionary.shape[2]
        self.k = dictionary.shape[1]
        self.vsa_conversion = vsa_conversion
        self.vsa_selection = vsa_selection
        self.context_superposition = context_superposition
        self.vsa_converter = VSAConverter(
            device, self.constellation, dictionary, dictionary_type="Continuous"
        )
        self.num_rules = num_rules
        self.rules_set = RulesSet(
            model=self.model,
            hidden_layers=hidden_layers,
            num_rules=self.num_rules,
            d_in=2 * self.d,
            d_out=-1,
            d_vsa=self.d,
            k=self.k,
            context_superpostion=self.context_superposition,
            context_keys=None,
            program=self.program,
            rule_type=self.rule_type,
            num_terms=self.num_terms,
            num_panels=self.num_panels,
        )

    def forward(self, scene_prob, targets=None, distribute=False):
        # convert logprob to VSAs
        scene_vsa = self.vsa_converter(scene_prob)
        # flatten scene
        scene = {}
        for attr in scene_vsa._fields:
            if attr in ["position", "number"] and not distribute:
                scene[attr] = None
            else:
                scene[attr] = t.flatten(
                    getattr(scene_vsa, attr),
                    start_dim=len(getattr(scene_vsa, attr).shape) - 2,
                )
        scene = type(scene_vsa)(**scene)
        # set indices for test panels
        if self.num_panels == 10:
            test_indeces = [9, 19]
        elif self.num_panels == 3:
            test_indeces = [2, 5]
        elif self.num_panels == 5:
            test_indeces = [4, 9]
        # compute output vectors
        output = dict()
        tests = dict()
        candidates = dict()
        for attr in scene._fields:
            if attr in ["position", "number"] and not distribute:
                tests[attr] = output[attr] = candidates[attr] = None
            else:
                tests[attr] = getattr(scene, attr)[:, test_indeces]
                output[attr] = self.rules_set(getattr(scene, attr))
                candidates[attr] = getattr(scene, attr)[:, -8:]
        # compile them in named tuples and return
        output = type(scene_vsa)(**output)
        tests = type(scene_vsa)(**tests)
        candidates = type(scene_vsa)(**candidates)
        return output, candidates, tests

    def anneal_softmax(self):
        for rule in self.rules_set.rules:
            rule.rule.anneal_softmax()


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
        num_panels=3,
    ):
        super(RulesSet, self).__init__()
        rule_class = GeneralRule
        if program:
            rules = ["add", "sub", "dist3", "progr"]
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
                    num_panels=num_panels,
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
        num_panels=3,
    ):
        super(GeneralRule, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d = d_vsa
        self.k = k
        self.context_superposition = context_superposition
        self.context_keys = context_keys
        if num_panels == 10:
            # I-RAVEN-X
            self.a3_indeces = range(0, 9)
            self.a6_indeces = range(10, 19)
            self.a9_indeces = range(20, 29)
            self.a3_context_indeces = [range(10, 20), range(20, 29)]
            self.a6_context_indeces = [range(0, 10), range(20, 29)]
            self.a9_context_indeces = [range(0, 10), range(10, 19)]
            self.rule = IravenxGeneralLearnableFormula(
                examples_len=9,
                context_len=19,
                k=self.k,
                num_terms=num_terms,
                program_rule=program_rule,
            )
        elif num_panels == 5:
            # I-RAVEN-V
            self.a3_indeces = range(0, 4)
            self.a6_indeces = range(5, 9)
            self.a9_indeces = range(10, 14)
            self.a3_context_indeces = [range(5, 10), range(10, 14)]
            self.a6_context_indeces = [range(0, 5), range(10, 14)]
            self.a9_context_indeces = [range(0, 5), range(5, 9)]
            self.rule = IravenVGeneralLearnableFormula(
                examples_len=4,
                context_len=9,
                k=self.k,
                num_terms=num_terms,
                program_rule=program_rule,
            )
        elif num_panels == 3 and num_terms == 12:
            # I-RAVEN
            self.a3_indeces = [0, 1]
            self.a6_indeces = [3, 4]
            self.a9_indeces = [6, 7]
            self.a3_context_indeces = [[3, 4, 5], [6, 7]]
            self.a6_context_indeces = [[0, 1, 2], [6, 7]]
            self.a9_context_indeces = [[0, 1, 2], [3, 4]]
            self.rule = ExtendedGeneralLearnableFormula(
                examples_len=2, context_len=5, k=self.k, program_rule=program_rule
            )

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
