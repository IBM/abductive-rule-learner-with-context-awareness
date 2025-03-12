#!/bin/bash

CONFIG="center_single"
SIGMA=0.7
RUN="arlc_learn_noisy_$SIGMA"
EXP_DIR="models"
EPOCHS=15
NTEST=5
NRULES=5
DATA="iravenx"
if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi



for SEED in $(seq 1 $NTEST);
do
    echo $SEED
    python main.py  --epochs $EPOCHS --n 3 --dyn_range 10 --num_terms 12 \
    --vsa_conversion --vsa_selection --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --partition _shuffle --entropy \
    --batch_size 8 --num_workers 1 --num_rules $NRULES --seed $SEED --run_name $RUN --exp_dir $EXP_DIR --orientation-confounder 0 --sigma $SIGMA
done
python results.py --path $EXP_DIR/$RUN --seeds $NTEST
