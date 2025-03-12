#!/bin/bash

MODEL=$1

CONFIG="center_single"
EXP_DIR="models"
EPOCHS=10
NTEST=5
NRULES=5
DEBUG=1
DATA="iraven"
SIGMA=0.3

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi

for SEED in $(seq 1 $NTEST);
do
    python main.py  --resume models/$MODEL/$SEED/ckpt  --n 3 --dyn_range 10 --num_terms 12 \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR   \
    --batch_size 512 --num_workers 1 --num_rules $NRULES --num_terms 12 --seed $SEED --run_name $MODEL --orientation-confounder 0 \
    --exp_dir $EXP_DIR --partition _shuffle --mode test --evaluate-rule --entropy  --sigma $SIGMA
done
python results.py --path models/$MODEL --seeds $NTEST
