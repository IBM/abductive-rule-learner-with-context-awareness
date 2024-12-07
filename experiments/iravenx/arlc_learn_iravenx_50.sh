#!/bin/bash

CONFIG="center_single"
RUN="iravenx_50"
EXP_DIR="models"
EPOCHS=10
NTEST=5
NRULES=5
DATA="iravenx"

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="path_top_standard_raven"
else
    DATA_DIR="path_to_iravenx"
fi

# Train the model
python main.py  --epochs $EPOCHS --dyn_range 50 --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 2 --run_name $RUN --exp_dir $EXP_DIR  --partition _shuffle

# Eval on unseen dynamic ranges
python main.py  --epochs $EPOCHS --dyn_range 100 --mode test --resume models/iravenx_50/2/ckpt --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 0 --run_name $RUN --exp_dir $EXP_DIR  --partition _shuffle

python main.py  --epochs $EPOCHS --dyn_range 1000 --mode test --resume models/iravenx_50/2/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS \
    --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 0 --run_name $RUN --exp_dir $EXP_DIR  --partition _shuffle



# Test arithmetic accuracies
python main.py  --epochs $EPOCHS --dyn_range 50 --mode test --resume models/iravenx_50/2/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS \
    --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 0 --run_name $RUN --exp_dir $EXP_DIR  --partition Arithmetic_shuffle

python main.py  --epochs $EPOCHS --dyn_range 100 --mode test --resume models/iravenx_50/2/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS \
    --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 0 --run_name $RUN --exp_dir $EXP_DIR  --partition Arithmetic_shuffle

python main.py  --epochs $EPOCHS --dyn_range 1000 --mode test --resume models/iravenx_50/2/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --annealing $EPOCHS \
    --batch_size 256 --num_workers 1 --num_rules $NRULES --num_terms 22 --seed 0 --run_name $RUN --exp_dir $EXP_DIR  --partition Arithmetic_shuffle