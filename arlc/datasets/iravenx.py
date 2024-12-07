# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import os
import numpy as np
import torch as t
from torch.utils.data import Dataset
from tqdm import tqdm
import json

rule_map = {
    "Constant": 0, 
    "Progression": 1, 
    "Arithmetic": 2, 
    "Distribute_Three": 3
}

class IRAVENXDataset(Dataset):
    def __init__(self,dataset_type,data_dir,constellation_filter,rule_filter="",attribute_filter="",n_train=None,in_memory=False,partition="",n=10, n_show=3, maxval=1000):
        
        self.n = n 
        self.n_show = n_show
        self.n_tot = n_show*n-1+8
        

        if dataset_type == "train": self.filtered_indeces = np.arange(6000)
        elif dataset_type == "val": self.filtered_indeces = np.arange(6000, 8000)
        elif dataset_type == "test": self.filtered_indeces = np.arange(8000, 10000)

        if rule_filter != "" or attribute_filter != "": raise ValueError("Rule filtering not implemented")        
        
        if n_train: self.filtered_indeces = self.filtered_indeces[:n_train]

        # load entire dataset from 
        with open("{}/{}{}_n_{}_maxval_{}.json".format(data_dir, constellation_filter, partition, n, maxval), "r") as f: self.dataset = json.load(f)
  

    def __len__(self): return len(self.filtered_indeces)

    def __getitem__(self, index):
        valid_index = self.filtered_indeces[index % len(self.filtered_indeces)]
        
        data = self.dataset[str(valid_index)]
        
        input_tensor = t.ones((self.n_tot, 9, 5)).float()*(-1) # dimension panel, slots, attributes
        input_tensor[:,0, 0] = 0 # Fix position in center constellation
        
        for i in range(self.n_tot):
            panel = data["rpm"][i+(self.n-self.n_show)*self.n][0]
            input_tensor[i, 0, 2] = int(panel["Color"])
            input_tensor[i, 0, 3] = int(panel["Size"]) + 1
            input_tensor[i, 0, 4] = int(panel["Type"])
            input_tensor[i, 0, 1] = int(panel["Angle"])
                    
        label_tensor = t.tensor(int(data["target"])).long()
        
        rules = data["rules"][0]
        # import pdb; pdb.set_trace()
        pos_num_rule = t.tensor(np.array(rule_map[rules["Number/Position"]])).float()
        color_rule = t.tensor(np.array(rule_map[rules["Color"]])).float()
        size_rule = t.tensor(np.array(rule_map[rules["Size"]])).float()
        type_rule = t.tensor(np.array(rule_map[rules["Type"]])).float()
        rules_tensor = t.stack([pos_num_rule, color_rule, size_rule, type_rule])
        return input_tensor, label_tensor, rules_tensor

