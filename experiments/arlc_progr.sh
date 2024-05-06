#!/bin/bash

python  main.py --vsa_conversion --vsa_selection  --shared_rules --config center_single  --data_dir data --batch_size 256 --num_workers 1  --mode test   --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config distribute_four  --data_dir data --batch_size 256 --num_workers 1   --mode test --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config distribute_nine  --data_dir data --batch_size 256 --num_workers 1   --mode test --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config left_right  --data_dir data --batch_size 256 --num_workers 1  --mode test  --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config up_down  --data_dir data --batch_size 256 --num_workers 1  --mode test  --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config in_out_single  --data_dir data --batch_size 256 --num_workers 1  --mode test  --program
python  main.py --vsa_conversion --vsa_selection  --shared_rules --config in_out_four  --data_dir data --batch_size 256 --num_workers 1  --mode test  --program
