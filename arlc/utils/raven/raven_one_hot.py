# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import torch

def smooth_dist(q):
    l = torch.argmax(q)
    temp = torch.arange(0, q.shape[0])
    sigma = 0.1
    temp = torch.div(torch.abs(temp - l),-2* sigma)
    temp = torch.exp(temp)
    temp = temp / temp.sum()
    return temp

def create_one_hot(puzzle, panel_constellation):
    eps = 10 ** (-10)
    batch_size = puzzle.shape[0]

    if panel_constellation == "center_single":
        exist_prob = torch.ones((batch_size,16,1,2)) * eps
        type_prob = torch.ones((batch_size,16,1,5)) * eps
        size_prob = torch.ones((batch_size,16,1,6)) * eps
        color_prob = torch.ones((batch_size,16,1,10)) * eps

        exist_prob[:,:,0,1] = 1

        for bs in range(batch_size):
            for i in range(16):
                exist_prob[bs,i,0]= smooth_dist(exist_prob[bs,i,0])
                type_prob[bs,i,0,int(puzzle[bs,i,0,4])]= 1
                size_prob[bs,i,0,int(puzzle[bs,i,0,3])]= 1
                color_prob[bs,i,0,int(puzzle[bs,i,0,2])]= 1
        return torch.log(exist_prob),torch.log(type_prob),torch.log(size_prob),torch.log(color_prob)

    if panel_constellation == "distribute_four":
        exist_prob = torch.ones((batch_size,16,4,2)) * eps
        type_prob = torch.ones((batch_size,16,4,5)) * eps
        size_prob = torch.ones((batch_size,16,4,6)) * eps
        color_prob = torch.ones((batch_size,16,4,10)) * eps

        for bs in range(batch_size):
            for i in range(16):
                temp = [0,1,2,3]
                for j in range(4):
                    if puzzle[bs,i,j,0] == -1:
                        k = temp[0]
                        exist_prob[bs,i,k,0] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs,i,j,0])
                        exist_prob[bs,i,k,1] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        type_prob[bs,i,k,int(puzzle[bs,i,j,4])]= 1
                        type_prob[bs,i,k]= smooth_dist(type_prob[bs,i,k])
                        size_prob[bs,i,k,int(puzzle[bs,i,j,3])]= 1
                        size_prob[bs,i,k]= smooth_dist(size_prob[bs,i,k])
                        color_prob[bs,i,k,int(puzzle[bs,i,j,2])]= 1
                        color_prob[bs,i,k]= smooth_dist(color_prob[bs,i,k])
                        temp.remove(k)
        return torch.log(exist_prob),torch.log(type_prob),torch.log(size_prob),torch.log(color_prob)

    if panel_constellation == "distribute_nine":
        exist_prob = torch.ones((batch_size,16,9,2)) * eps
        type_prob = torch.ones((batch_size,16,9,5)) * eps
        size_prob = torch.ones((batch_size,16,9,6)) * eps
        color_prob = torch.ones((batch_size,16,9,10)) * eps

        for bs in range(batch_size):
            for i in range(16):
                temp = [0,1,2,3,4,5,6,7,8]
                for j in range(9):
                    if puzzle[bs,i,j,0] == -1:
                        k = temp[0]
                        exist_prob[bs,i,k,0] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs,i,j,0])
                        exist_prob[bs,i,k,1] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        type_prob[bs,i,k,int(puzzle[bs,i,j,4])]= 1
                        type_prob[bs,i,k]= smooth_dist(type_prob[bs,i,k])
                        size_prob[bs,i,k,int(puzzle[bs,i,j,3])]= 1
                        size_prob[bs,i,k]= smooth_dist(size_prob[bs,i,k])
                        color_prob[bs,i,k,int(puzzle[bs,i,j,2])]= 1
                        color_prob[bs,i,k]= smooth_dist(color_prob[bs,i,k])
                        temp.remove(k)
        return torch.log(exist_prob),torch.log(type_prob),torch.log(size_prob),torch.log(color_prob)
      
    if panel_constellation == "left_right" or panel_constellation=="up_down" or panel_constellation == "in_out_single":
        exist_prob = torch.ones((batch_size,16,2,2)) * eps
        type_prob = torch.ones((batch_size,16,2,5)) * eps
        size_prob = torch.ones((batch_size,16,2,6)) * eps
        color_prob = torch.ones((batch_size,16,2,10)) * eps

        exist_prob[:,:,0,1] = 1
        exist_prob[:,:,1,1] = 1

        for bs in range(batch_size):
            for i in range(16):
                for j in range(2):
                    k = int(puzzle[bs,i,j,0])
                    exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                    type_prob[bs,i,k,int(puzzle[bs,i,j,4])]= 1
                    size_prob[bs,i,k,int(puzzle[bs,i,j,3])]= 1
                    color_prob[bs,i,k,int(puzzle[bs,i,j,2])]= 1
        return torch.log(exist_prob),torch.log(type_prob),torch.log(size_prob),torch.log(color_prob)
    
    if panel_constellation == "in_out_four":
        exist_prob = torch.ones((batch_size,16,5,2)) * eps
        type_prob = torch.ones((batch_size,16,5,5)) * eps
        size_prob = torch.ones((batch_size,16,5,6)) * eps
        color_prob = torch.ones((batch_size,16,5,10)) * eps

        for bs in range(batch_size):
            for i in range(16):
                temp = [0,1,2,3,4]
                for j in range(5):
                    if puzzle[bs,i,j,0] == -1:
                        k = temp[0]
                        exist_prob[bs,i,k,0] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        temp.remove(k)
                    else:
                        k = int(puzzle[bs,i,j,0])
                        exist_prob[bs,i,k,1] = 1
                        exist_prob[bs,i,k]= smooth_dist(exist_prob[bs,i,k])
                        type_prob[bs,i,k,int(puzzle[bs,i,j,4])]= 1
                        type_prob[bs,i,k]= smooth_dist(type_prob[bs,i,k])
                        size_prob[bs,i,k,int(puzzle[bs,i,j,3])]= 1
                        size_prob[bs,i,k]= smooth_dist(size_prob[bs,i,k])
                        color_prob[bs,i,k,int(puzzle[bs,i,j,2])]= 1
                        color_prob[bs,i,k]= smooth_dist(color_prob[bs,i,k])
                        temp.remove(k)
        return torch.log(exist_prob),torch.log(type_prob),torch.log(size_prob),torch.log(color_prob)