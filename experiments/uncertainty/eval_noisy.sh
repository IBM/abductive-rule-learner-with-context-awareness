#!/bin/bash

RUN=$1
SIGMA=$2

CONFIG="center_single"
EXP_DIR="models"
EPOCHS=10
NTEST=5
NRULES=5
DATA="iravenx"
N_CONFOUNDERS=0

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi

echo "****************** EVAL 100"
for SEED in $(seq 1 $NTEST);
do
    python main.py --dyn_range 100 --n 10  --resume models/$RUN/$SEED/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR  \
    --batch_size 128 --num_workers 1 --num_rules $NRULES --num_terms 26 --seed $SEED --run_name $RUN \
    --exp_dir $EXP_DIR --partition _shuffle --mode test --evaluate-rule --entropy  --sigma $SIGMA
done
python results.py --path $EXP_DIR/$RUN --seeds $NTEST


echo "****************** EVAL 1000"
for SEED in $(seq 1 $NTEST);
do
    python main.py --dyn_range 1000 --n 10  --resume models/$RUN/$SEED/ckpt \
    --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR  \
    --batch_size 128 --num_workers 1 --num_rules $NRULES --num_terms 26 --seed $SEED --run_name $RUN \
    --exp_dir $EXP_DIR --partition _shuffle --mode test --evaluate-rule --entropy  --sigma $SIGMA
done
python results.py --path $EXP_DIR/$RUN --seeds $NTEST
