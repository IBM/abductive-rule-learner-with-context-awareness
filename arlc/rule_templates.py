# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import torch as t
import torch.nn as nn
from nvsa.reasoning.vsa_block_utils import (
    block_binding2,
    block_binding3,
    block_unbinding2,
)

def oh(tensor, idx, val=10001.0):
    tensor[idx] = val
    return tensor


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_vsa, hidden_layers=3, softmax=False):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_vsa = d_vsa
        self.hidden_layers = hidden_layers

        if self.d_in != self.d_vsa:
            self.learnable_vsa = nn.LazyLinear(self.d_vsa)

        layers = []

        for _ in range(0, self.hidden_layers):
            layers.append(nn.LazyLinear(self.d_vsa))
            layers.append(nn.LayerNorm(self.d_vsa))
            layers.append(nn.ReLU())

        if softmax:
            layers.append(nn.Linear(self.d_vsa, self.d_out))
            layers.append(nn.LayerNorm(self.d_out))
            layers.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) > 2:
            if self.d_in != self.d_vsa:
                x = self.learnable_vsa(x)
            x = x.reshape((x.shape[0], -1))
        return self.layers(x)


class LearnableFormula(nn.Module):
    def __init__(self, d_vsa, k, context_indeces, hardcode=None):
        super(LearnableFormula, self).__init__()
        self.d_vsa = d_vsa
        self.k = k
        self.context_len = len(context_indeces) + 1
        self.hardcode = hardcode

        idx_term_map = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "i"]
        self.idx_term_map = [idx_term_map[i] for i in context_indeces].append("i")

        if hardcode is None:
            self.n1 = nn.Parameter(t.randn(self.context_len))
            self.n2 = nn.Parameter(t.randn(self.context_len))
            self.n3 = nn.Parameter(t.randn(self.context_len))
            self.d1 = nn.Parameter(t.randn(self.context_len))
            self.d2 = nn.Parameter(t.randn(self.context_len))
            self.d3 = nn.Parameter(t.randn(self.context_len))
        else:
            self.n1 = nn.Parameter(hardcode[0])
            self.n2 = nn.Parameter(hardcode[1])
            self.n3 = nn.Parameter(hardcode[2])
            self.d1 = nn.Parameter(hardcode[3])
            self.d2 = nn.Parameter(hardcode[4])
            self.d3 = nn.Parameter(hardcode[5])

        self.softmax = nn.Softmax(dim=-1)

    def __str__(self):
        cf = (
            nn.functional.one_hot(self.softmax(self.n1).argmax(), 9)
            + nn.functional.one_hot(self.softmax(self.n2).argmax(), 9)
            + nn.functional.one_hot(self.softmax(self.n3).argmax(), 9)
            - nn.functional.one_hot(self.softmax(self.d1).argmax(), 9)
            - nn.functional.one_hot(self.softmax(self.d2).argmax(), 9)
            - nn.functional.one_hot(self.softmax(self.d3).argmax(), 9)
        )
        terms = ["a11", "a12", "a13", "a21", "a22", "a23", "a31", "a32"]
        hr_rule = [
            f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]
        ]
        return "".join(hr_rule)

    def add_identity(self, x):
        identity = t.zeros_like(x[:, 0]).unsqueeze(1)
        identity[:, :, :, 0] = 1
        x_with_identity = t.cat((x, identity), dim=1)
        return x_with_identity

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.k, -1)
        x = self.add_identity(x)
        if self.hardcode is None:
            self.att_scores_n1 = self.softmax(self.n1.unsqueeze(0).unsqueeze(0))
            self.att_scores_n2 = self.softmax(self.n2.unsqueeze(0).unsqueeze(0))
            self.att_scores_n3 = self.softmax(self.n3.unsqueeze(0).unsqueeze(0))
            self.att_scores_d1 = self.softmax(self.d1.unsqueeze(0).unsqueeze(0))
            self.att_scores_d2 = self.softmax(self.d2.unsqueeze(0).unsqueeze(0))
            self.att_scores_d3 = self.softmax(self.d3.unsqueeze(0).unsqueeze(0))
        else:
            self.att_scores_n1 = self.n1.unsqueeze(0).unsqueeze(0)
            self.att_scores_n2 = self.n2.unsqueeze(0).unsqueeze(0)
            self.att_scores_n3 = self.n3.unsqueeze(0).unsqueeze(0)
            self.att_scores_d1 = self.d1.unsqueeze(0).unsqueeze(0)
            self.att_scores_d2 = self.d2.unsqueeze(0).unsqueeze(0)
            self.att_scores_d3 = self.d3.unsqueeze(0).unsqueeze(0)
        x = x.view(x.shape[0], x.shape[1], -1)
        n1 = (
            t.matmul(self.att_scores_n1.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        n2 = (
            t.matmul(self.att_scores_n2.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        n3 = (
            t.matmul(self.att_scores_n3.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        d1 = (
            t.matmul(self.att_scores_d1.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        d2 = (
            t.matmul(self.att_scores_d2.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        d3 = (
            t.matmul(self.att_scores_d3.repeat(x.shape[0], 1, 1), x)
            .squeeze(1)
            .view(x.shape[0], self.k, -1)
        )
        n = block_binding3(n1, n2, n3)
        d = block_binding3(d1, d2, d3)
        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output


class GeneralLearnableFormula(nn.Module):
    def __init__(self, examples_len, context_len, k):
        super(GeneralLearnableFormula, self).__init__()
        self.k = k
        self.context_len = context_len
        self.examples_len = examples_len

        self.n1 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.n2 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.n3 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.d1 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.d2 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.d3 = nn.Parameter(t.randn(examples_len + context_len + 1))

        self.softmax = nn.Softmax(dim=-1)

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

        self.att_scores_n1 = self.softmax(self.n1.unsqueeze(0).unsqueeze(0))
        self.att_scores_n2 = self.softmax(self.n2.unsqueeze(0).unsqueeze(0))
        self.att_scores_n3 = self.softmax(self.n3.unsqueeze(0).unsqueeze(0))
        self.att_scores_d1 = self.softmax(self.d1.unsqueeze(0).unsqueeze(0))
        self.att_scores_d2 = self.softmax(self.d2.unsqueeze(0).unsqueeze(0))
        self.att_scores_d3 = self.softmax(self.d3.unsqueeze(0).unsqueeze(0))

        n1 = t.matmul(self.att_scores_n1.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        n2 = t.matmul(self.att_scores_n2.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        n3 = t.matmul(self.att_scores_n3.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d1 = t.matmul(self.att_scores_d1.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d2 = t.matmul(self.att_scores_d2.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)
        d3 = t.matmul(self.att_scores_d3.repeat(x.shape[0], 1, 1), x).squeeze(1).view(x.shape[0], self.k, -1)

        n = block_binding3(n1, n2, n3)
        d = block_binding3(d1, d2, d3)
        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output

    def __str__(self):
        cf = nn.functional.one_hot(self.softmax(self.n1).argmax(), self.examples_len + self.context_len + 1)+ nn.functional.one_hot(self.softmax(self.n2).argmax(), self.examples_len + self.context_len + 1)+ nn.functional.one_hot(self.softmax(self.n3).argmax(), self.examples_len + self.context_len + 1)- nn.functional.one_hot(self.softmax(self.d1).argmax(), self.examples_len + self.context_len + 1)- nn.functional.one_hot(self.softmax(self.d2).argmax(), self.examples_len + self.context_len + 1)- nn.functional.one_hot(self.softmax(self.d3).argmax(), self.examples_len + self.context_len + 1)
        # dist = [
        #     self.softmax(self.n1).tolist(),
        #     self.softmax(self.n2).tolist(),
        #     self.softmax(self.n3).tolist(),
        #     self.softmax(self.d1).tolist(),
        #     self.softmax(self.d2).tolist(),
        #     self.softmax(self.d3).tolist(),
        # ]

        nterms = self.context_len + self.examples_len
        terms = zip(range(nterms), ["e1", "e2", "c1", "c2", "c3", "c4", "c5", "c6"])
        hr_rule = "".join([f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in terms if cf[i]])
        # for d in dist:
        #     hr_rule += "\n" + " ".join(['{:.2f}'.format(x) for x in d])
        return hr_rule


class ExtendedGeneralLearnableFormula(nn.Module):
    def __init__(self, examples_len, context_len, k, program_rule=None):
        super(ExtendedGeneralLearnableFormula, self).__init__()
        self.k = k
        self.context_len = context_len
        self.examples_len = examples_len
        self.program_rule = program_rule
        if program_rule:self.program_weights(rule=program_rule)
        else:
            self.n1 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.n2 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.n3 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.n4 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.n5 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.n6 = nn.Parameter(t.randn(examples_len + context_len + 1))

            self.d1 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.d2 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.d3 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.d4 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.d5 = nn.Parameter(t.randn(examples_len + context_len + 1))
            self.d6 = nn.Parameter(t.randn(examples_len + context_len + 1))
        self.T = nn.Parameter(t.tensor(1.0), requires_grad=False)
        self.softmax = nn.Softmax(dim=-1)

    def annealing_update(self, epoch, max_epoch):
        if max_epoch > epoch:
            self.T.data = self.T.data - (self.T.data - 0.01) / max_epoch * epoch

    def program_weights(self, rule):
        identity = t.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000.0])
        if rule == "constant":
            self.n1 = nn.Parameter(t.tensor([0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n2 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0]))
            self.n3 = nn.Parameter(identity)
            self.n4 = nn.Parameter(identity)
            self.n5 = nn.Parameter(identity)
            self.n6 = nn.Parameter(identity)
            self.d1 = nn.Parameter(t.tensor([0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d2 = nn.Parameter(identity)
            self.d3 = nn.Parameter(identity)
            self.d4 = nn.Parameter(identity)
            self.d5 = nn.Parameter(identity)
            self.d6 = nn.Parameter(identity)
        elif rule == "add":
            self.n1 = nn.Parameter(t.tensor([1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n2 = nn.Parameter(t.tensor([0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n3 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0]))
            self.n4 = nn.Parameter(identity)
            self.n5 = nn.Parameter(identity)
            self.n6 = nn.Parameter(identity)
            self.d1 = nn.Parameter(t.tensor([0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d2 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0]))
            self.d4 = nn.Parameter(identity)
            self.d5 = nn.Parameter(identity)
            self.d3 = nn.Parameter(identity)
            self.d6 = nn.Parameter(identity)
        elif rule == "sub":
            self.n1 = nn.Parameter(t.tensor([1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n2 = nn.Parameter(identity)
            self.n3 = nn.Parameter(identity)
            self.n4 = nn.Parameter(identity)
            self.n5 = nn.Parameter(identity)
            self.n6 = nn.Parameter(identity)
            self.d1 = nn.Parameter(t.tensor([0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d2 = nn.Parameter(identity)
            self.d4 = nn.Parameter(identity)
            self.d5 = nn.Parameter(identity)
            self.d3 = nn.Parameter(identity)
            self.d6 = nn.Parameter(identity)
        elif rule == "dist3":
            self.n1 = nn.Parameter(t.tensor([0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n2 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0]))
            self.n3 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0]))
            self.n4 = nn.Parameter(t.tensor([0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.n5 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0]))
            self.n6 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0]))
            self.d1 = nn.Parameter(t.tensor([1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d2 = nn.Parameter(t.tensor([0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d3 = nn.Parameter(identity)
            self.d4 = nn.Parameter(t.tensor([0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            self.d5 = nn.Parameter(t.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1000000.0, 0.0, 0.0]))
            self.d6 = nn.Parameter(t.tensor([1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        elif rule == "bonus":
            self.n1 = nn.Parameter(identity);self.n2 = nn.Parameter(identity);self.n3 = nn.Parameter(identity);self.n4 = nn.Parameter(identity);self.n5 = nn.Parameter(identity);self.n6 = nn.Parameter(identity)
            self.d1 = nn.Parameter(identity);self.d2 = nn.Parameter(identity);self.d4 = nn.Parameter(identity);self.d5 = nn.Parameter(identity);self.d3 = nn.Parameter(identity);self.d6 = nn.Parameter(identity)

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
            term = t.matmul(attn_score.repeat(x.shape[0], 1, 1), input).squeeze(1).view(input.shape[0], self.k, -1)
            return term

        n1 = wcomb(self.n1, x)
        n2 = wcomb(self.n2, x)
        n3 = wcomb(self.n3, x)
        n4 = wcomb(self.n4, x)
        n5 = wcomb(self.n5, x)
        n6 = wcomb(self.n6, x)

        d1 = wcomb(self.d1, x)
        d2 = wcomb(self.d2, x)
        d3 = wcomb(self.d3, x)
        d4 = wcomb(self.d4, x)
        d5 = wcomb(self.d5, x)
        d6 = wcomb(self.d6, x)

        n = block_binding3(
            block_binding2(n1, n2), block_binding2(n3, n4), block_binding2(n5, n6)
        )
        d = block_binding3(
            block_binding2(d1, d2), block_binding2(d3, d4), block_binding2(d5, d6)
        )

        output = block_unbinding2(n, d)
        output = output.view(output.shape[0], -1)
        return output

    def __str__(self):
        tl = self.context_len + self.examples_len + 1
        cf = (
            nn.functional.one_hot(self.softmax(self.n1).argmax(), tl)
            + nn.functional.one_hot(self.softmax(self.n2).argmax(), tl)
            + nn.functional.one_hot(self.softmax(self.n3).argmax(), tl)
            + nn.functional.one_hot(self.softmax(self.n4).argmax(), tl)
            + nn.functional.one_hot(self.softmax(self.n5).argmax(), tl)
            + nn.functional.one_hot(self.softmax(self.n6).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d1).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d2).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d3).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d4).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d5).argmax(), tl)
            - nn.functional.one_hot(self.softmax(self.d6).argmax(), tl)
        )
        terms = ["x1", "x2", "c1", "c2", "c3", "c4", "c5"]
        hr_rule = [
            f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]
        ]
        return "".join(hr_rule)

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
        self.T = t.tensor(1.0)

    
    def init_terms(self, num_terms, num_panels):
        terms = list()
        for _ in range(num_terms): 
            terms.append(nn.Parameter(t.randn(num_panels)))
            # terms.append(nn.Parameter(oh(t.zeros(num_panels), -1, 1)))
        # for i in range(12, 13):
        #     terms[i].data = oh(t.zeros(num_panels), i-12, 1)    # -x1 -x2 ... -x9
        self.terms = nn.ParameterList(terms)


    def annealing_update(self, epoch, max_epoch):
        if max_epoch > epoch:
            self.T.data = self.T.data - (self.T.data - 0.01) / max_epoch * epoch
         
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
            attn_score = self.softmax(weights.unsqueeze(0).unsqueeze(0)/self.T)
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

        n = bind_seq([wcomb(t, x) for t in self.terms[ : len(self.terms)//2]])
        d = bind_seq([wcomb(t, x) for t in self.terms[len(self.terms)//2 : ]])

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
            self.terms[0] = oh(self.terms[0], 0)    # +x1
            self.terms[1] = oh(self.terms[0], 0)    # +x1
            self.terms[12] = oh(self.terms[0], 0)   # -x1
            
        elif rule == "add":
            for i in range(9):
                self.terms[i] = oh(t.zeros(29, device=device), i)  # +x1 +x2 ... +x9
            self.terms[9] = oh(t.zeros(29, device=device), 0)      # +x1
            self.terms[12] = oh(t.zeros(29, device=device), 0)     # -x1
            
        elif rule == "sub":
            for i in range(12, 21):
                self.terms[i] = oh(t.zeros(29, device=device), i-12)   # -x1 -x2 ... -x9
            self.terms[0] = oh(t.zeros(29, device=device), 0)          # +x1
            self.terms[1] = oh(t.zeros(29, device=device), 0)          # +x1

        elif rule == "dist3":
            for i in range(10):self.terms[i] = oh(t.zeros(29, device=device), 9+i)  # +c1 +c2 ... +c10
            for i in range(13, 22):self.terms[i] = oh(t.zeros(29, device=device), i-13)   # -x1 -x2 ... -x9

        elif rule == "progr":
            self.terms[0] = oh(t.zeros(29, device=device), 1)    # +x2
            self.terms[1] = oh(t.zeros(29, device=device), 8)    # +x9
            self.terms[12] = oh(t.zeros(29, device=device), 0)   # -x1

    def __str__(self):
        tl = self.context_len + self.examples_len + 1
        cfp = nn.functional.one_hot(self.softmax(self.terms[0]/self.T).argmax(), tl)
        # add + terms
        for i in range(1, len(self.terms)//2):cfp += nn.functional.one_hot(self.softmax(self.terms[i]/self.T).argmax(), tl)

        cfm = - nn.functional.one_hot(self.softmax(self.terms[len(self.terms)//2]/self.T).argmax(), tl)
        # add - terms
        for i in range(len(self.terms)//2+1, len(self.terms)): cfm -= nn.functional.one_hot(self.softmax(self.terms[i]/self.T).argmax(), tl)

        cf = cfp + cfm
        terms = [f"x{i+1}" for i in range(self.examples_len)] + [f"c{i+1}" for i in range(self.context_len)] + ["e"]
        hr_rule = [f"{'+' if cf[i]>0 else ''}{cf[i]}{x} " for i, x in enumerate(terms) if cf[i]]
        # hr_ruleplus = [
        #     f"+{cfp[i]}{x} " for i, x in enumerate(terms) if cfp[i]
        # ]
        # hr_rulemin = [
        #     f"-{cfm[i]}{x} " for i, x in enumerate(terms) if cfm[i]
        # ]
        return "".join(hr_rule) #+ "\n" +  "".join(hr_ruleplus) + "".join(hr_rulemin) + "\n\n"
