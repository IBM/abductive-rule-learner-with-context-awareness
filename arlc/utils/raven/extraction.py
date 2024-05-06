# *----------------------------------------------------------------------------*
# * Copyright (C) 2024 IBM Inc. All rights reserved                            *
# * SPDX-License-Identifier: GPL-3.0-only                                      *
# *----------------------------------------------------------------------------*

import numpy as np; import os; import xml.etree.ElementTree as ET; import argparse; import tqdm

parser = argparse.ArgumentParser(description='NVSA'); parser.add_argument('--data_path', type=str, default = "/dccstor/saentis/data/I-RAVEN")
pos_num_rule_idx_map_four = {"Constant": 0, "Progression_One_Pos": 1, "Progression_Mone_Pos": 2, "Arithmetic_Plus_Pos": 3, "Arithmetic_Minus_Pos": 4, "Distribute_Three_Left_Pos": 5, "Distribute_Three_Right_Pos": 6, "Progression_One_Num": 7, "Progression_Mone_Num": 8, "Arithmetic_Plus_Num": 9, "Arithmetic_Minus_Num": 10, "Distribute_Three_Left_Num": 11, "Distribute_Three_Right_Num": 12}
pos_num_rule_idx_map_nine = {"Constant": 0, "Progression_One_Pos": 1, "Progression_Mone_Pos": 2, "Progression_Two_Pos": 3, "Progression_Mtwo_Pos": 4, "Arithmetic_Plus_Pos": 5, "Arithmetic_Minus_Pos": 6, "Distribute_Three_Left_Pos": 7, "Distribute_Three_Right_Pos": 8, "Progression_One_Num": 9, "Progression_Mone_Num": 10, "Progression_Two_Num": 11, "Progression_Mtwo_Num": 12, "Arithmetic_Plus_Num": 13, "Arithmetic_Minus_Num": 14, "Distribute_Three_Left_Num": 15, "Distribute_Three_Right_Num": 16}
type_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Distribute_Three_Left": 5, "Distribute_Three_Right": 6}
size_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Arithmetic_Plus": 5, "Arithmetic_Minus": 6, "Distribute_Three_Left": 7, "Distribute_Three_Right": 8}
color_rule_idx_map = {"Constant": 0, "Progression_One": 1, "Progression_Mone": 2, "Progression_Two": 3, "Progression_Mtwo": 4, "Arithmetic_Plus": 5, "Arithmetic_Minus": 6, "Distribute_Three_Left": 7, "Distribute_Three_Right": 8}

def get_pos_num_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][0].attrib["name"]
    attrib_name = xml_rules[rule_idx][0].attrib["attr"][:3]
    if index_name == "Progression":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second = int(xml_panels[1][0][comp_idx][0].attrib["Number"])
            if second == first + 1:
                index_name += "_One_Num"
            if second == first - 1:
                index_name += "_Mone_Num"
            if second == first + 2:
                index_name += "_Two_Num"
            if second == first - 2:
                index_name += "_Mtwo_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second = []
            for entity in xml_panels[1][0][comp_idx][0]:
                second.append(all_position.index(eval(entity.attrib["bbox"])))
            third = []
            for entity in xml_panels[2][0][comp_idx][0]:
                third.append(all_position.index(eval(entity.attrib["bbox"])))
            fourth = []
            for entity in xml_panels[3][0][comp_idx][0]:
                fourth.append(all_position.index(eval(entity.attrib["bbox"])))
            fifth = []
            for entity in xml_panels[4][0][comp_idx][0]:
                fifth.append(all_position.index(eval(entity.attrib["bbox"])))
            sixth = []
            for entity in xml_panels[5][0][comp_idx][0]:
                sixth.append(all_position.index(eval(entity.attrib["bbox"])))
            seventh = []
            for entity in xml_panels[6][0][comp_idx][0]:
                seventh.append(all_position.index(eval(entity.attrib["bbox"])))
            eighth = []
            for entity in xml_panels[7][0][comp_idx][0]:
                eighth.append(all_position.index(eval(entity.attrib["bbox"])))
            if len(set(map(lambda index: (index + 1) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index + 1) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_One_Pos"
            if len(set(map(lambda index: (index - 1) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index - 1) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Mone_Pos"
            if len(set(map(lambda index: (index + 2) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index + 2) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Two_Pos"
            if len(set(map(lambda index: (index - 2) % num_elements, first)) - set(second)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, second)) - set(third)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, fourth)) - set(fifth)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, fifth)) - set(sixth)) == 0 and \
               len(set(map(lambda index: (index - 2) % num_elements, seventh)) - set(eighth)) == 0:
                index_name += "_Mtwo_Pos"
            if index_name.endswith("_One_Pos_Mone_Pos"):
                if np.random.uniform() >= 0.5:
                    index_name = "Progression_One_Pos"
                else:
                    index_name = "Progression_Mone_Pos"
    if index_name == "Arithmetic":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second = int(xml_panels[1][0][comp_idx][0].attrib["Number"])
            third = int(xml_panels[2][0][comp_idx][0].attrib["Number"])
            if third == first + second + 1:
                index_name += "_Plus_Num"
            if third == first - second - 1:
                index_name += "_Minus_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second = []
            for entity in xml_panels[1][0][comp_idx][0]:
                second.append(all_position.index(eval(entity.attrib["bbox"])))
            third = []
            for entity in xml_panels[2][0][comp_idx][0]:
                third.append(all_position.index(eval(entity.attrib["bbox"])))
            if set(third) == set(first).union(set(second)):
                index_name += "_Plus_Pos"
            if set(third) == set(first) - set(second):
                index_name += "_Minus_Pos"
    if index_name == "Distribute_Three":
        if attrib_name == "Num":
            first = int(xml_panels[0][0][comp_idx][0].attrib["Number"])
            second_left = int(xml_panels[5][0][comp_idx][0].attrib["Number"])
            second_right = int(xml_panels[4][0][comp_idx][0].attrib["Number"])
            if second_left == first:
                index_name += "_Left_Num"
            if second_right == first:
                index_name += "_Right_Num"
        if attrib_name == "Pos":
            all_position = eval(xml_panels[0][0][comp_idx][0].attrib["Position"])
            first = []
            for entity in xml_panels[0][0][comp_idx][0]:
                first.append(all_position.index(eval(entity.attrib["bbox"])))
            second_left = []
            for entity in xml_panels[5][0][comp_idx][0]:
                second_left.append(all_position.index(eval(entity.attrib["bbox"])))
            second_right = []
            for entity in xml_panels[4][0][comp_idx][0]:
                second_right.append(all_position.index(eval(entity.attrib["bbox"])))
            if set(second_left) == set(first):
                index_name += "_Left_Pos"
            if set(second_right) == set(first):
                index_name += "_Right_Pos"
    return pos_num_rule_idx_map[index_name]


def get_type_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][1].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Type"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Type"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Type"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Type"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Type"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return type_rule_idx_map[index_name]


def get_size_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][2].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Size"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Arithmetic":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Size"])
        third = int(xml_panels[2][0][comp_idx][0][0].attrib["Size"])
        if third == first + second + 1:
            index_name += "_Plus"
        if third == first - second - 1:
            index_name += "_Minus"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Size"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Size"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Size"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return size_rule_idx_map[index_name]

def get_color_rule(rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules):
    index_name = xml_rules[rule_idx][3].attrib["name"]
    if index_name == "Progression":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Color"])
        if second == first + 1:
            index_name += "_One"
        if second == first - 1:
            index_name += "_Mone"
        if second == first + 2:
            index_name += "_Two"
        if second == first - 2:
            index_name += "_Mtwo"
    if index_name == "Arithmetic":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second = int(xml_panels[1][0][comp_idx][0][0].attrib["Color"])
        third = int(xml_panels[2][0][comp_idx][0][0].attrib["Color"])
        fourth = int(xml_panels[3][0][comp_idx][0][0].attrib["Color"])
        fifth = int(xml_panels[4][0][comp_idx][0][0].attrib["Color"])
        sixth = int(xml_panels[5][0][comp_idx][0][0].attrib["Color"])
        if (third == first + second) and (sixth == fourth + fifth):
            index_name += "_Plus"
        if (third == first - second) and (sixth == fourth - fifth):
            index_name += "_Minus"
    if index_name == "Distribute_Three":
        first = int(xml_panels[0][0][comp_idx][0][0].attrib["Color"])
        second_left = int(xml_panels[5][0][comp_idx][0][0].attrib["Color"])
        second_right = int(xml_panels[4][0][comp_idx][0][0].attrib["Color"])
        if second_left == first:
            index_name += "_Left"
        if second_right == first:
            index_name += "_Right"
    return color_rule_idx_map[index_name]
    
def main():
    args = parser.parse_args()
    DATA_PATH = args.data_path
    constellation_name_list = ["center_single", "distribute_four", "distribute_nine", "left_center_single_right_center_single", "up_center_single_down_center_single", "in_center_single_out_center_single", "in_distribute_four_out_center_single"]
    save_name_list = ["center_single_extracted_with_rules", "distribute_four_extracted_with_rules", "distribute_nine_extracted_with_rules", "left_right_extracted_with_rules", "up_down_extracted_with_rules","in_out_single_extracted_with_rules","in_out_four_extracted_with_rules"] 
    obj_name_list = ["train", "val", "test"]
    my_bbox = {"in_out_four_extracted_with_rules":{"[0.5, 0.5, 1, 1]":0, "[0.42, 0.42, 0.15, 0.15]":1, "[0.42, 0.58, 0.15, 0.15]":2,"[0.58, 0.42, 0.15, 0.15]":3,"[0.58, 0.58, 0.15, 0.15]":4}, "in_out_single_extracted_with_rules":{ "[0.5, 0.5, 1, 1]":0, "[0.5, 0.5, 0.33, 0.33]":1}, "up_down_extracted_with_rules": { "[0.25, 0.5, 0.5, 0.5]":0, "[0.75, 0.5, 0.5, 0.5]":1},"left_right_extracted_with_rules": {"[0.5, 0.25, 0.5, 0.5]":0, "[0.5, 0.75, 0.5, 0.5]":1}, "distribute_nine_extracted_with_rules": {"[0.16, 0.16, 0.33, 0.33]":0, "[0.16, 0.5, 0.33, 0.33]":1,"[0.16, 0.83, 0.33, 0.33]":2, "[0.5, 0.16, 0.33, 0.33]":3, "[0.5, 0.5, 0.33, 0.33]":4,"[0.5, 0.83, 0.33, 0.33]":5,"[0.83, 0.16, 0.33, 0.33]":6,"[0.83, 0.5, 0.33, 0.33]":7, "[0.83, 0.83, 0.33, 0.33]":8},"distribute_four_extracted_with_rules": {"[0.25, 0.25, 0.5, 0.5]":0, "[0.25, 0.75, 0.5, 0.5]":1, "[0.75, 0.25, 0.5, 0.5]":2, "[0.75, 0.75, 0.5, 0.5]":3} }
    for w in range(len(constellation_name_list)):
        file_type = constellation_name_list[w]
        save_name = save_name_list[w]
        path = os.path.join(DATA_PATH,save_name)
        path_train, path_val, path_test = os.path.join(path, "train"), os.path.join(path, "val"), os.path.join(path, "test")
        os.makedirs(path,exist_ok=True); os.makedirs(path_train,exist_ok=True); os.makedirs(path_val,exist_ok=True); os.makedirs(path_test,exist_ok=True)
        for n in range(len(obj_name_list)):
            count = 0
            obj_name = obj_name_list[n]
            for j in tqdm.tqdm(range(10001)):
                try:
                    tree= ET.parse('{0}/{1}/RAVEN_{2}_{3}.xml'.format(DATA_PATH,file_type,j, obj_name))
                except:
                    continue
                root = tree.getroot()
                xml_panels = root[0]
                xml_rules = root[1]
                rule_idx = 0
                comp_idx = 0
                num_elements = 9
                pos_num_rule_idx_map = pos_num_rule_idx_map_four
                if file_type == 'distribute_four':
                    num_elements = 4
                    pos_num_rule_idx_map = pos_num_rule_idx_map_four
                elif file_type == 'distribute_nine':
                    num_elements = 9
                    pos_num_rule_idx_map = pos_num_rule_idx_map_nine
                args = [rule_idx, comp_idx, num_elements, pos_num_rule_idx_map, xml_panels, xml_rules]
                pos_num_rule = np.array(get_pos_num_rule(*args))
                color_rule = np.array(get_color_rule(*args))
                size_rule = np.array(get_size_rule(*args))
                type_rule = np.array(get_type_rule(*args))
                rules = np.array([pos_num_rule, color_rule, size_rule, type_rule])
                idx_panel= 0
                for panel in root[0]:
                    idx = 0
                    for component in panel [0]:
                        for entity in component[0]:
                            a = entity.attrib
                            angle, color, size, typ, bbox = int(a.get('Angle')), int(a.get('Color')), int(a.get('Size')), int(a.get('Type'))-1, a.get('bbox')
                            pos = my_bbox[save_name][bbox] if save_name!="center_single_extracted_with_rules" else 0
                            ext_comp = [pos,angle,color,size,typ]
                            ext_comp = np.expand_dims(ext_comp,axis=0)
                            ext_panel = ext_comp if (idx==0) else np.concatenate((ext_panel,ext_comp),axis = 0)
                            idx = idx + 1
                    c = 9 - idx
                    if c >0:
                        filler = np.ones((c,5)) * (-1)
                        ext_panel = np.concatenate((ext_panel,filler),axis = 0)
                    ext_panel = np.expand_dims(ext_panel,axis=0)
                    ext_sample = ext_panel if (idx_panel == 0) else np.concatenate((ext_sample,ext_panel),axis = 0)
                    idx_panel = idx_panel + 1
                file= np.load('{0}/{1}/RAVEN_{2}_{3}.npz'.format(DATA_PATH,file_type,j, obj_name))
                filename = "{0}/{1}/{2}/RAVEN_{3}_{2}.npz".format(DATA_PATH,save_name,obj_name,count)
                np.savez (filename, target = file['target'], predict = file['predict'], image = file['image'], meta_matrix = file['meta_matrix'], meta_structure= file['meta_structure'],
                    meta_target= file['meta_target'], structure= file['structure'] , extracted_meta = ext_sample, rules=rules)
                count = count + 1
        print("finished with: ", save_name)
        
if __name__ == '__main__':
    main()
