#!/bin/bash

CONFIG="center_single"
CONF=5
RUN="iravenx_confounders_entropy_$CONF"
EXP_DIR="models"
NTEST=5
NRULES=5
DEBUG=0
DATA="iravenx"
EPOCHS=10

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi


for SEED in $(seq 1 $NTEST);
do
    echo $SEED
    python main.py  --epochs $EPOCHS --dyn_range 100 --n 10 \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR --entropy \
    --batch_size 8 --num_workers 1 --num_rules $NRULES --num_terms 26 --seed $SEED --run_name $RUN --exp_dir $EXP_DIR  --partition _shuffle --orientation-confounder $CONF
done
python results.py --path $EXP_DIR/$RUN --seeds $NTEST
