# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import torch as t
import torch.nn as nn
from nvsa.reasoning.vsa_block_utils import (
    block_binding2,
    block_unbinding2,
)


def oh(tensor, idx, val=10001.0):
    tensor[idx] = val
    return tensor


class ExtendedGeneralLearnableFormula(nn.Module):
    def __init__(self, examples_len, context_len, k, program_rule=None):
        super(ExtendedGeneralLearnableFormula, self).__init__()
        self.k = k
        self.context_len = context_len
        self.examples_len = examples_len
        self.program_rule = program_rule
        if program_rule:
            self.program_weights(rule=program_rule)
        else:
            self.init_terms(12, examples_len + context_len + 1)
        self.softmax = nn.Softmax(dim=-1)
        self.T = 1

    def init_terms(self, num_terms, num_panels):
        terms = list()
        for _ in range(num_terms):
            terms.append(nn.Parameter(t.randn(num_panels)))
        self.terms = nn.ParameterList(terms)

    def program_weights(self, rule, device="cuda"):
        print(f"Programming {rule}")
        # init every term with the identity
        self.terms = []
        for i in range(12):
            self.terms.append(oh(t.zeros(8, device=device), -1))
        if rule == "add":
            self.terms[0] = oh(t.zeros(8, device=device), 0)  # +x1
            self.terms[1] = oh(t.zeros(8, device=device), 1)  # +x2
            self.terms[2] = oh(t.zeros(8, device=device), 4)  # +c3
            self.terms[7] = oh(t.zeros(8, device=device), 2)  # -c1
            self.terms[8] = oh(t.zeros(8, device=device), 3)  # -c2
        elif rule == "sub":
            self.terms[0] = oh(t.zeros(8, device=device), 0)  # +x1
            self.terms[6] = oh(t.zeros(8, device=device), 1)  # -x2
            self.terms[2] = oh(t.zeros(8, device=device), 2)  # +c1
            self.terms[7] = oh(t.zeros(8, device=device), 3)  # -c2
            self.terms[8] = oh(t.zeros(8, device=device), 4)  # -c3
        elif rule == "dist3":
            self.terms[0] = oh(t.zeros(8, device=device), 2)  # +c1
            self.terms[1] = oh(t.zeros(8, device=device), 3)  # +c2
            self.terms[2] = oh(t.zeros(8, device=device), 4)  # +c3
            self.terms[6] = oh(t.zeros(8, device=device), 0)  # -x1
            self.terms[7] = oh(t.zeros(8, device=device), 1)  # -x2
        elif rule == "progr":
            self.terms[0] = oh(t.zeros(8, device=device), 1)  # +x2
            self.terms[1] = oh(t.zeros(8, device=device), 1)  # +x2
            self.terms[6] = oh(t.zeros(8, device=device), 0)  # -x1

    def add_identity(self, x):
        identity = t.zeros_like(x[:, 0]).unsqueeze(1)
        identity[:, :, :, 0] = 1
        x_with_identity = t.cat((x, identity), dim=1)
        return x_with_identity

    def forward(self, x, ctx):
        x = t.cat([x, ctx], dim=1)
        x = x.reshape(x.shape[0], x.shape[1], self.k, -1)
        x = self.add_identity(x)
        x = x.view(x.shape[0], x.shape[1], -1)

        def wcomb(weights, input):
            attn_score = self.softmax(weights.unsqueeze(0).unsqueeze(0) / self.T)
            term = (
                t.matmul(attn_score.repeat(input.shape[0], 1, 1), input)
                .squeeze(1)
                .view(input.shape[0], self.k, -1)
            )
            return term

        def bind_seq(seq):
            seq_len = len(seq)
            res = seq[0]
            for i in range(1, seq_len):
                res = block_binding2(res, seq[i])
            return res

        n = bind_seq([wcomb(t, x) for t in self.terms[: len(self.terms) // 2]])
        d = bind_seq([wcomb(t, x) for t in self.terms[len(self.terms) // 2 :]])
        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output

    def __str__(self):
        tl = self.context_len + self.examples_len + 1
        cfp = nn.functional.one_hot(self.softmax(self.terms[0]).argmax(), tl)
        # add + terms
        for i in range(1, len(self.terms) // 2):
            cfp += nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)

        cfm = -nn.functional.one_hot(
            self.softmax(self.terms[len(self.terms) // 2]).argmax(), tl
        )
        # add - terms
        for i in range(len(self.terms) // 2 + 1, len(self.terms)):
            cfm -= nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)

        cf = cfp + cfm

        terms = [f"x{i+1}" for i in range(self.examples_len)] + [
            f"c{i+1}" for i in range(self.context_len)
        ]
        hr_rule = [
            f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]
        ]
        return "".join(hr_rule)

    def anneal_softmax(self):
        self.T = 0.01


class IravenxGeneralLearnableFormula(nn.Module):
    def __init__(self, examples_len, context_len, k, num_terms=12, program_rule=None):
        super(IravenxGeneralLearnableFormula, self).__init__()
        self.k = k
        self.context_len = context_len
        self.examples_len = examples_len
        self.program_rule = program_rule
        self.num_terms = num_terms
        if program_rule:
            self.program_weights(rule=program_rule, num_terms=num_terms)
        else:
            self.init_terms(num_terms, examples_len + context_len + 1)
        self.softmax = nn.Softmax(dim=-1)

    def init_terms(self, num_terms, num_panels):
        terms = list()
        for _ in range(num_terms):
            terms.append(nn.Parameter(t.randn(num_panels)))
            # terms.append(nn.Parameter(oh(t.zeros(num_panels), -1, 1)))
        # for i in range(12, 13):
        #     terms[i].data = oh(t.zeros(num_panels), i-12, 1)    # -x1 -x2 ... -x9
        self.terms = nn.ParameterList(terms)

    def add_identity(self, x):
        identity = t.zeros_like(x[:, 0]).unsqueeze(1)
        identity[:, :, :, 0] = 1
        x_with_identity = t.cat((x, identity), dim=1)
        return x_with_identity

    def forward(self, x, ctx):
        x = t.cat([x, ctx], dim=1)
        x = x.reshape(x.shape[0], x.shape[1], self.k, -1)
        x = self.add_identity(x)
        x = x.view(x.shape[0], x.shape[1], -1)

        def wcomb(weights, input):
            attn_score = self.softmax(weights.unsqueeze(0).unsqueeze(0))
            term = (
                t.matmul(attn_score.repeat(input.shape[0], 1, 1), input)
                .squeeze(1)
                .view(input.shape[0], self.k, -1)
            )
            return term

        def bind_seq(seq):
            seq_len = len(seq)
            res = seq[0]
            for i in range(1, seq_len):
                res = block_binding2(res, seq[i])
            return res

        n = bind_seq([wcomb(t, x) for t in self.terms[: len(self.terms) // 2]])
        d = bind_seq([wcomb(t, x) for t in self.terms[len(self.terms) // 2 :]])

        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output

    def program_weights(self, rule, num_terms, device="cuda"):
        print(f"Programming {rule}")

        # init every term with the identity
        self.terms = []
        for i in range(num_terms):
            self.terms.append(oh(t.zeros(29, device=device), -1))

        if rule == "constant":
            self.terms[0] = oh(self.terms[0], 0)  # +x1
            self.terms[1] = oh(self.terms[0], 0)  # +x1
            self.terms[12] = oh(self.terms[0], 0)  # -x1

        elif rule == "add":
            for i in range(9):
                self.terms[i] = oh(t.zeros(29, device=device), i)  # +x1 +x2 ... +x9
            self.terms[9] = oh(t.zeros(29, device=device), 0)  # +x1
            self.terms[12] = oh(t.zeros(29, device=device), 0)  # -x1

        elif rule == "sub":
            for i in range(12, 21):
                self.terms[i] = oh(
                    t.zeros(29, device=device), i - 12
                )  # -x1 -x2 ... -x9
            self.terms[0] = oh(t.zeros(29, device=device), 0)  # +x1
            self.terms[1] = oh(t.zeros(29, device=device), 0)  # +x1

        elif rule == "dist3":
            for i in range(10):
                self.terms[i] = oh(
                    t.zeros(29, device=device), 9 + i
                )  # +c1 +c2 ... +c10
            for i in range(13, 22):
                self.terms[i] = oh(
                    t.zeros(29, device=device), i - 13
                )  # -x1 -x2 ... -x9

        elif rule == "progr":
            self.terms[0] = oh(t.zeros(29, device=device), 1)  # +x2
            self.terms[1] = oh(t.zeros(29, device=device), 8)  # +x9
            self.terms[12] = oh(t.zeros(29, device=device), 0)  # -x1

    def __str__(self):
        tl = self.context_len + self.examples_len + 1
        cfp = nn.functional.one_hot(self.softmax(self.terms[0]).argmax(), tl)
        # add + terms
        for i in range(1, len(self.terms) // 2):
            cfp += nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)

        cfm = -nn.functional.one_hot(
            self.softmax(self.terms[len(self.terms) // 2]).argmax(), tl
        )
        # add - terms
        for i in range(len(self.terms) // 2 + 1, len(self.terms)):
            cfm -= nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)

        cf = cfp + cfm

        terms = (
            [f"x{i+1}" for i in range(self.examples_len)]
            + [f"c{i+1}" for i in range(self.context_len)]
            + ["e"]
        )
        hr_rule = [
            f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]
        ]
        return "".join(
            hr_rule
        )


class IravenVGeneralLearnableFormula(nn.Module):
    def __init__(self, examples_len, context_len, k, num_terms=12, program_rule=None):
        super(IravenVGeneralLearnableFormula, self).__init__()
        self.k = k
        self.context_len = context_len
        self.examples_len = examples_len
        self.program_rule = program_rule
        self.num_terms = num_terms
        if program_rule:
            self.program_weights(rule=program_rule, num_terms=num_terms)
        else:
            self.init_terms(num_terms, examples_len + context_len + 1)
        self.softmax = nn.Softmax(dim=-1)

    def init_terms(self, num_terms, num_panels):
        terms = list()
        for _ in range(num_terms):
            terms.append(nn.Parameter(t.randn(num_panels)))
            # terms.append(nn.Parameter(oh(t.zeros(num_panels), -1, 1)))
        # for i in range(12, 13):
        #     terms[i].data = oh(t.zeros(num_panels), i-12, 1)    # -x1 -x2 ... -x9
        self.terms = nn.ParameterList(terms)

    def add_identity(self, x):
        identity = t.zeros_like(x[:, 0]).unsqueeze(1)
        identity[:, :, :, 0] = 1
        x_with_identity = t.cat((x, identity), dim=1)
        return x_with_identity

    def forward(self, x, ctx):
        x = t.cat([x, ctx], dim=1)
        x = x.reshape(x.shape[0], x.shape[1], self.k, -1)
        x = self.add_identity(x)
        x = x.view(x.shape[0], x.shape[1], -1)

        def wcomb(weights, input):
            attn_score = self.softmax(weights.unsqueeze(0).unsqueeze(0))
            term = (
                t.matmul(attn_score.repeat(input.shape[0], 1, 1), input)
                .squeeze(1)
                .view(input.shape[0], self.k, -1)
            )
            return term

        def bind_seq(seq):
            seq_len = len(seq)
            res = seq[0]
            for i in range(1, seq_len):
                res = block_binding2(res, seq[i])
            return res

        n = bind_seq([wcomb(t, x) for t in self.terms[: len(self.terms) // 2]])
        d = bind_seq([wcomb(t, x) for t in self.terms[len(self.terms) // 2 :]])

        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output

    def program_weights(self, rule, num_terms, device="cuda"):
        print(f"Programming {rule}")

        # init every term with the identity
        self.terms = []
        for i in range(num_terms):
            self.terms.append(oh(t.zeros(14, device=device), -1))

        if rule == "constant":
            self.terms[0] = oh(self.terms[0], 0)  # +x1
            self.terms[1] = oh(self.terms[0], 0)  # +x1
            self.terms[11] = oh(self.terms[0], 0)  # -x1

        elif rule == "add":
            for i in range(4):
                self.terms[i] = oh(t.zeros(14, device=device), i)  # +x1 +x2 ... +x9
            self.terms[4] = oh(t.zeros(14, device=device), 0)  # +x1
            self.terms[11] = oh(t.zeros(14, device=device), 0)  # -x1

        elif rule == "sub":
            for i in range(10, 14):
                self.terms[i] = oh(
                    t.zeros(14, device=device), i - 10
                )  # -x1 -x2 ... -x9
            self.terms[0] = oh(t.zeros(14, device=device), 0)  # +x1
            self.terms[1] = oh(t.zeros(14, device=device), 0)  # +x1

        elif rule == "dist3":
            for i in range(5):
                self.terms[i] = oh(
                    t.zeros(14, device=device), 4 + i
                )  # +c1 +c2 ... +c10
            for i in range(10, 14):
                self.terms[i] = oh(
                    t.zeros(14, device=device), i - 10
                )  # -x1 -x2 ... -x9

        elif rule == "progr":
            self.terms[0] = oh(t.zeros(14, device=device), 1)  # +x2
            self.terms[1] = oh(t.zeros(14, device=device), 3)  # +x3
            self.terms[10] = oh(t.zeros(14, device=device), 0)  # -x1

    def __str__(self):
        tl = self.context_len + self.examples_len + 1
        cfp = nn.functional.one_hot(self.softmax(self.terms[0]).argmax(), tl)
        # add + terms
        for i in range(1, len(self.terms) // 2):
            cfp += nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)
        cfm = -nn.functional.one_hot(
            self.softmax(self.terms[len(self.terms) // 2]).argmax(), tl
        )
        # add - terms
        for i in range(len(self.terms) // 2 + 1, len(self.terms)):
            cfm -= nn.functional.one_hot(self.softmax(self.terms[i]).argmax(), tl)
        cf = cfp + cfm
        terms = [f"x{i+1}" for i in range(self.examples_len)] + [
            f"c{i+1}" for i in range(self.context_len)
        ]
        hr_rule = [
            f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]
        ]
        return "".join(
            hr_rule
        )
