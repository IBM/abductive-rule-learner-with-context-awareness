#!/bin/bash

CONFIG="center_single"
EXP_DIR="models"
NTEST=5
NRULES=5
DATA="iravenx"
EPOCHS=10
SIGMA=0.7
RUN="iravenx_noisy_$SIGMA"

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi

for SEED in $(seq 1 $NTEST);
do
    echo $SEED
    python main.py  --epochs $EPOCHS --dyn_range 100 --n 10 \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --entropy  --sigma $SIGMA \
    --batch_size 8 --num_workers 1 --num_rules $NRULES --num_terms 26 --seed $SEED --run_name $RUN --exp_dir $EXP_DIR  --partition _shuffle
done
python results.py --path $EXP_DIR/$RUN --seeds $NTEST
