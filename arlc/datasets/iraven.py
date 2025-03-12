# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import os
import numpy as np
import torch as t
from torch.utils.data import Dataset
import random
import json

rule_map = {"Constant": 0, "Progression": 1, "Arithmetic": 2, "Distribute_Three": 3}


class GeneralIRAVENDataset(Dataset):
    def __init__(
        self,
        dataset_type,
        data_dir,
        constellation_filter,
        rule_filter="",
        attribute_filter="",
        n_train=None,
        in_memory=False,
        partition="",
        n=10,
        n_show=3,
        maxval=1000,
        n_confounders=0,
    ):

        self.n = n
        self.n_show = n_show
        self.n_tot = n_show * n - 1 + 8
        self.n_confounders = n_confounders
        self.maxval = maxval

        if dataset_type == "train":
            self.filtered_indeces = np.arange(6000)
        elif dataset_type == "val":
            self.filtered_indeces = np.arange(6000, 8000)
        elif dataset_type == "test":
            self.filtered_indeces = np.arange(8000, 10000)

        if rule_filter != "" or attribute_filter != "":
            raise ValueError("Rule filtering not implemented")

        if n_train:
            self.filtered_indeces = self.filtered_indeces[:n_train]

        self.old_raven = not "I-RAVEN-" in data_dir
        data_file = (
            f"{constellation_filter}{partition}_n_{n}_maxval_{maxval}.json"
        )
        print(f"Number of confounders: {n_confounders}")
        self.constellation = constellation_filter
        # load entire dataset from
        with open(os.path.join(data_dir, data_file), "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.filtered_indeces)

    def _get_panel_number(self, x, y):
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Point is outside the 1x1 box")
        if self.constellation == "distribute_nine":
            div = 1 / 3
            ppr = 3
        elif self.constellation == "distribute_four":
            div = 1 / 2
            ppr = 2
        col = int(x / div)
        row = int((1 - y) / div)
        panel_number = row * ppr + col
        return panel_number

    def __getitem__(self, index):
        valid_index = self.filtered_indeces[index % len(self.filtered_indeces)]
        data = self.dataset[str(valid_index)]
        # dimension panel, slots, attributes
        input_tensor = t.ones((self.n_tot, 9, 5 + self.n_confounders)).float() * (-1)
        for i in range(self.n_tot):
            panels = data["rpm"][i + (self.n - self.n_show) * self.n][0]
            if self.constellation == "center_single":
                input_tensor[:, 0, 0] = 0  # Fix position in center constellation
                panel = data["rpm"][i + (self.n - self.n_show) * self.n][0]
                input_tensor[i, 0, 2] = int(panel["Color"])
                input_tensor[i, 0, 3] = int(panel["Size"]) + self.old_raven * 1
                input_tensor[i, 0, 4] = int(panel["Type"]) + self.old_raven * 2
                input_tensor[i, 0, 1] = int(panel["Angle"])
                for n in range(self.n_confounders):
                    input_tensor[i, 0, 5 + n] = int(panel[f"Confounder{n}"])
            else:
                for pidx, (pos, ent) in enumerate(
                    zip(panels["positions"], panels["entities"])
                ):
                    input_tensor[i, pidx, 2] = int(ent["Color"])
                    input_tensor[i, pidx, 3] = int(ent["Size"])
                    input_tensor[i, pidx, 4] = int(ent["Type"])
                    input_tensor[i, pidx, 1] = int(ent["Angle"])
                    input_tensor[i, pidx, 0] = int(self._get_panel_number(*pos[:2]))
                    for n in range(self.n_confounders):
                        input_tensor[i, 0, 5 + n] = random.randint(0, self.maxval)

        label_tensor = t.tensor(int(data["target"])).long()
        rules = data["rules"][0]
        if "Number/Position" in rules:
            num_pos = "Number/Position"
        elif "Number" in rules:
            num_pos = "Number"
        else:
            num_pos = "Position"
        pos_num_rule = t.tensor(np.array(rule_map[rules[num_pos]])).float()
        color_rule = t.tensor(np.array(rule_map[rules["Color"]])).float()
        size_rule = t.tensor(np.array(rule_map[rules["Size"]])).float()
        type_rule = t.tensor(np.array(rule_map[rules["Type"]])).float()
        rules_tensor = t.stack([pos_num_rule, color_rule, size_rule, type_rule])
        return input_tensor, label_tensor, rules_tensor


if __name__ == "__main__":
    dataset = GeneralIRAVENDataset(
        "train",
        "/dccstor/saentis/data/I-RAVEN-X",
        "center_single",
        n=10,
        maxval=100,
        partition="_shuffle",
        n_confounders=10,
    )
    print(dataset.__getitem__(0))
