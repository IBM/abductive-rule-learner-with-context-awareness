# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import os
import numpy as np
import torch as t
from torch.utils.data import Dataset

class RAVENDataset(Dataset):
    
    def __init__(self, dataset_type,data_dir, constellation_filter,rule_filter = "", attribute_filter = "", n_train = None):
        self.constellation_filter = constellation_filter
        self.extracted_path = os.path.join(data_dir, constellation_filter + "_extracted_with_rules", dataset_type)
        self.dataset_type = dataset_type
        self.filtered_indeces = self.filter_dataset(attribute_filter, rule_filter)
        if n_train: self.filtered_indeces = self.filtered_indeces[:n_train]
        
    def rule_idx(self, rule_name, attribute_name):
        if attribute_name == "Position":
            if self.constellation_filter == "distribute_four":
                rule_idx_map = {"Constant": [0], "Progression": [1, 2], "Arithmetic": [3, 4], "Distribute_Three": [5, 6]}
            if self.constellation_filter == "distribute_nine":
                rule_idx_map = {"Constant": [0], "Progression": [9, 10, 11, 12], "Arithmetic": [13, 14], "Distribute_Three": [15, 16]}
        elif attribute_name == "Number":
            if self.constellation_filter == "distribute_four":
                rule_idx_map = {"Constant": [0], "Progression": [7, 8], "Arithmetic": [9, 10], "Distribute_Three": [11, 12]}
            if self.constellation_filter == "distribute_nine":
                rule_idx_map = {"Constant": [0], "Progression": [9, 10, 11, 12], "Arithmetic": [13, 14], "Distribute_Three": [15, 16]}
        elif attribute_name == "Type":
            rule_idx_map = {"Constant": [0], "Progression": [1, 2, 3, 4], "Distribute_Three": [5, 6]}
        elif attribute_name == "Size":
            rule_idx_map = {"Constant": [0], "Progression": [1, 2, 3, 4], "Arithmetic": [5, 6], "Distribute_Three": [7, 8]}
        elif attribute_name == "Color":
            rule_idx_map = {"Constant": [0], "Progression": [1, 2, 3, 4], "Arithmetic": [5, 6], "Distribute_Three": [7, 8]}
        return rule_idx_map[rule_name]

    def filter_dataset(self, attribute_filter, rule_filter):
        attribute_idx = {"Position": 0, "Number": 0, "Color": 1, "Size": 2, "Type": 3}
        filtered_indeces = []
        num_indeces = len(os.listdir(self.extracted_path))
        if attribute_filter == "" or rule_filter == "": filtered_indeces = list(range(num_indeces))
        else:
            for index in range(num_indeces):
                data_path = os.path.join(self.extracted_path, "RAVEN_" + str(index) + "_" + self.dataset_type + ".npz")
                data = np.load(data_path)
                rules = data["rules"]
                if self.dataset_type == "train" or self.dataset_type == "val":
                    if rules[attribute_idx[attribute_filter]] not in self.rule_idx(rule_filter, attribute_filter): filtered_indeces.append(index)
                else: 
                    if rules[attribute_idx[attribute_filter]] in self.rule_idx(rule_filter, attribute_filter): filtered_indeces.append(index)
        return filtered_indeces

    def __len__(self):
        return len(self.filtered_indeces)

    def __getitem__(self, index):
        valid_index = self.filtered_indeces[index%len(self.filtered_indeces)]
        data_path = os.path.join(self.extracted_path, "RAVEN_" + str(valid_index) + "_" + self.dataset_type + ".npz")
        data = np.load(data_path)
        input_tensor = t.from_numpy(data["extracted_meta"]).float()
        label_tensor = t.from_numpy(data["target"]).long()
        rules = data["rules"]
        pos_num_rule = t.from_numpy(np.array(rules[0])).float()
        color_rule = t.from_numpy(np.array(rules[1])).float()
        size_rule = t.from_numpy(np.array(rules[2])).float()
        type_rule = t.from_numpy(np.array(rules[3])).float()
        rules_tensor = t.stack([pos_num_rule, color_rule, size_rule, type_rule])
        return input_tensor, label_tensor, rules_tensor