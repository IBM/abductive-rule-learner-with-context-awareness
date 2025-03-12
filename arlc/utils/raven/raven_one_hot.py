# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import torch
from arlc.utils.const import DIM_ONEHOT


def smooth_dist(q, sigma=0.1):
    if sigma < 0:
        return bin_dist(q, -sigma)
    elif sigma == 0:
        return q
    l = torch.argmax(q)
    temp = torch.arange(0, q.shape[0])
    temp = torch.div(torch.abs(temp - l), -2 * sigma)
    temp = torch.exp(temp)
    temp = temp / temp.sum()
    return temp


def bin_dist(q, threshold):
    """Smooth a probability distribution by binning.
    In practice, the method takes a tensor representing a PMF and returns a tensor where
    the probability of the most likely value PT is sampled in [q, 1], and the probability of
    its two neighbours PN1 and PN2 are sampled in [0, 1-q] and 1-PT-PN1, respectively.

    Args:
        q (torch.Tensor): input PMF tensor
    """
    l = torch.argmax(q)
    q[l] = threshold + (1 - threshold) * torch.rand(1)
    right_index = (l + 1) % q.shape[0]
    left_index = (l - 1) % q.shape[0]
    q[right_index] = (1 - q[l]) * torch.rand(1)
    q[left_index] = 1 - q[l] - q[right_index]
    return q / q.sum()


def create_one_hot(puzzle, panel_constellation, sigma=0.1):
    eps = 10 ** (-10)
    batch_size, num_panels, _, num_att = puzzle.shape

    if panel_constellation == "center_single":
        exist_prob = torch.ones((batch_size, num_panels, 1, 2)) * eps
        type_prob = torch.ones((batch_size, num_panels, 1, DIM_ONEHOT)) * eps
        size_prob = torch.ones((batch_size, num_panels, 1, DIM_ONEHOT)) * eps
        color_prob = torch.ones((batch_size, num_panels, 1, DIM_ONEHOT)) * eps
        angle_prob = torch.ones((batch_size, num_panels, 1, DIM_ONEHOT)) * eps
        confounders_prob = [
            torch.ones((batch_size, num_panels, 1, DIM_ONEHOT)) * eps
            for _ in range(num_att - 5)
        ]
        exist_prob[:, :, 0, 1] = 1
        for bs in range(batch_size):
            for i in range(num_panels):
                exist_prob[bs, i, 0] = smooth_dist(exist_prob[bs, i, 0], sigma)
                type_prob[bs, i, 0, int(puzzle[bs, i, 0, 4])] = 1
                type_prob[bs, i, 0] = smooth_dist(type_prob[bs, i, 0], sigma)
                size_prob[bs, i, 0, int(puzzle[bs, i, 0, 3])] = 1
                size_prob[bs, i, 0] = smooth_dist(size_prob[bs, i, 0], sigma)
                color_prob[bs, i, 0, int(puzzle[bs, i, 0, 2])] = 1
                color_prob[bs, i, 0] = smooth_dist(color_prob[bs, i, 0], sigma)
                angle_prob[bs, i, 0, int(puzzle[bs, i, 0, 1])] = 1
                angle_prob[bs, i, 0] = smooth_dist(angle_prob[bs, i, 0], sigma)
                for j in range(len(confounders_prob)):
                    confounders_prob[j][bs, i, 0, int(puzzle[bs, i, 0, 5 + j])] = 1
                    confounders_prob[j][bs, i, 0] = smooth_dist(
                        confounders_prob[j][bs, i, 0], sigma
                    )
        att_prob = {
            "exist": torch.log(exist_prob),
            "type": torch.log(type_prob),
            "size": torch.log(size_prob),
            "color": torch.log(color_prob),
            "angle": torch.log(angle_prob),
        }
        conf_prob = {
            f"confounder{i}": torch.log(confounders_prob[i])
            for i in range(len(confounders_prob))
        }
        return {**att_prob, **conf_prob}

    if panel_constellation == "distribute_four":
        exist_prob = torch.ones((batch_size, num_panels, 4, 2)) * eps
        type_prob = torch.ones((batch_size, num_panels, 4, DIM_ONEHOT)) * eps
        size_prob = torch.ones((batch_size, num_panels, 4, DIM_ONEHOT)) * eps
        color_prob = torch.ones((batch_size, num_panels, 4, DIM_ONEHOT)) * eps
        angle_prob = torch.ones((batch_size, num_panels, 4, DIM_ONEHOT)) * eps

        for bs in range(batch_size):
            for i in range(num_panels):
                temp = [0, 1, 2, 3]
                for j in range(4):
                    if puzzle[bs, i, j, 0] == -1:
                        k = temp[0]
                        exist_prob[bs, i, k, 0] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs, i, j, 0])
                        exist_prob[bs, i, k, 1] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        type_prob[bs, i, k, int(puzzle[bs, i, j, 4])] = 1
                        type_prob[bs, i, k] = smooth_dist(type_prob[bs, i, k])
                        size_prob[bs, i, k, int(puzzle[bs, i, j, 3])] = 1
                        size_prob[bs, i, k] = smooth_dist(size_prob[bs, i, k])
                        color_prob[bs, i, k, int(puzzle[bs, i, j, 2])] = 1
                        color_prob[bs, i, k] = smooth_dist(color_prob[bs, i, k])
                        angle_prob[bs, i, k, int(puzzle[bs, i, j, 1])] = 1
                        angle_prob[bs, i, k] = smooth_dist(angle_prob[bs, i, k])
                        temp.remove(k)
        return {
            "exist": torch.log(exist_prob),
            "type": torch.log(type_prob),
            "size": torch.log(size_prob),
            "color": torch.log(color_prob),
            "angle": torch.log(angle_prob),
        }

    if panel_constellation == "distribute_nine":
        exist_prob = torch.ones((batch_size, num_panels, 9, 2)) * eps
        type_prob = torch.ones((batch_size, num_panels, 9, DIM_ONEHOT)) * eps
        size_prob = torch.ones((batch_size, num_panels, 9, DIM_ONEHOT)) * eps
        color_prob = torch.ones((batch_size, num_panels, 9, DIM_ONEHOT)) * eps
        angle_prob = torch.ones((batch_size, num_panels, 9, DIM_ONEHOT)) * eps

        for bs in range(batch_size):
            for i in range(num_panels):
                temp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                for j in range(9):
                    if puzzle[bs, i, j, 0] == -1:
                        k = temp[0]
                        exist_prob[bs, i, k, 0] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs, i, j, 0])
                        exist_prob[bs, i, k, 1] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        type_prob[bs, i, k, int(puzzle[bs, i, j, 4])] = 1
                        type_prob[bs, i, k] = smooth_dist(type_prob[bs, i, k])
                        size_prob[bs, i, k, int(puzzle[bs, i, j, 3])] = 1
                        size_prob[bs, i, k] = smooth_dist(size_prob[bs, i, k])
                        color_prob[bs, i, k, int(puzzle[bs, i, j, 2])] = 1
                        color_prob[bs, i, k] = smooth_dist(color_prob[bs, i, k])
                        angle_prob[bs, i, k, int(puzzle[bs, i, j, 1])] = 1
                        angle_prob[bs, i, k] = smooth_dist(angle_prob[bs, i, k])
                        temp.remove(k)
        return {
            "exist": torch.log(exist_prob),
            "type": torch.log(type_prob),
            "size": torch.log(size_prob),
            "color": torch.log(color_prob),
            "angle": torch.log(angle_prob),
        }

    if (
        panel_constellation == "left_right"
        or panel_constellation == "up_down"
        or panel_constellation == "in_out_single"
    ):
        exist_prob = torch.ones((batch_size, num_panels, 2, 2)) * eps
        type_prob = torch.ones((batch_size, num_panels, 2, DIM_ONEHOT)) * eps
        size_prob = torch.ones((batch_size, num_panels, 2, DIM_ONEHOT)) * eps
        color_prob = torch.ones((batch_size, num_panels, 2, DIM_ONEHOT)) * eps
        angle_prob = torch.ones((batch_size, num_panels, 2, DIM_ONEHOT)) * eps
        exist_prob[:, :, 0, 1] = 1
        exist_prob[:, :, 1, 1] = 1

        for bs in range(batch_size):
            for i in range(num_panels):
                for j in range(2):
                    k = int(puzzle[bs, i, j, 0])
                    exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                    type_prob[bs, i, k, int(puzzle[bs, i, j, 4])] = 1
                    size_prob[bs, i, k, int(puzzle[bs, i, j, 3])] = 1
                    color_prob[bs, i, k, int(puzzle[bs, i, j, 2])] = 1
                    angle_prob[bs, i, k, int(puzzle[bs, i, j, 1])] = 1
        return (
            torch.log(exist_prob),
            torch.log(type_prob),
            torch.log(size_prob),
            torch.log(color_prob),
            torch.log(angle_prob),
        )

    if panel_constellation == "in_out_four":
        exist_prob = torch.ones((batch_size, num_panels, 5, 2)) * eps
        type_prob = torch.ones((batch_size, num_panels, 5, DIM_ONEHOT)) * eps
        size_prob = torch.ones((batch_size, num_panels, 5, DIM_ONEHOT)) * eps
        color_prob = torch.ones((batch_size, num_panels, 5, DIM_ONEHOT)) * eps
        angle_prob = torch.ones((batch_size, num_panels, 5, DIM_ONEHOT)) * eps

        for bs in range(batch_size):
            for i in range(num_panels):
                temp = [0, 1, 2, 3, 4]
                for j in range(5):
                    if puzzle[bs, i, j, 0] == -1:
                        k = temp[0]
                        exist_prob[bs, i, k, 0] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs, i, j, 0])
                        exist_prob[bs, i, k, 1] = 1
                        exist_prob[bs, i, k] = smooth_dist(exist_prob[bs, i, k])
                        type_prob[bs, i, k, int(puzzle[bs, i, j, 4])] = 1
                        type_prob[bs, i, k] = smooth_dist(type_prob[bs, i, k])
                        size_prob[bs, i, k, int(puzzle[bs, i, j, 3])] = 1
                        size_prob[bs, i, k] = smooth_dist(size_prob[bs, i, k])
                        color_prob[bs, i, k, int(puzzle[bs, i, j, 2])] = 1
                        color_prob[bs, i, k] = smooth_dist(color_prob[bs, i, k])
                        angle_prob[bs, i, k, int(puzzle[bs, i, j, 1])] = 1
                        angle_prob[bs, i, k] = smooth_dist(angle_prob[bs, i, k])
                        temp.remove(k)
        return (
            torch.log(exist_prob),
            torch.log(type_prob),
            torch.log(size_prob),
            torch.log(color_prob),
            torch.log(angle_prob),
        )
