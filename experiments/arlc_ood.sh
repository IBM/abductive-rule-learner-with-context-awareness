#!/bin/bash

python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Type'  --gen_rule 'Constant'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Type'  --gen_rule 'Progression'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Type'  --gen_rule 'Distribute_Three'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Size'  --gen_rule 'Constant'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Size'  --gen_rule 'Progression'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Size'  --gen_rule 'Distribute_Three'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Size'  --gen_rule 'Arithmetic'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Color' --gen_rule 'Constant'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Color' --gen_rule 'Progression'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Color' --gen_rule 'Distribute_Three'
python main.py --data_dir data --vsa_conversion --vsa_selection --shared_rules  --gen_attribute 'Color' --gen_rule 'Arithmetic'
