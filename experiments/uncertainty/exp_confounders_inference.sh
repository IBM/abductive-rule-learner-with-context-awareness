#!/bin/bash

RUN=$1

CONFIG="center_single"
EXP_DIR="models"
EPOCHS=10
NTEST=3
NRULES=5
DEBUG=1
DATA="iravenx"
N_CONFOUNDERS=(0 1 3 5 10 30 300)

if [ "$DATA" = "iraven" ]; then
    DATA_DIR="data/I-RAVEN"
else
    DATA_DIR="data/I-RAVEN-X"
fi

for N_CONFOUNDERS in  "${N_CONFOUNDERS[@]}";
do
    echo "****************** N_CONF $N_CONFOUNDERS"
    for SEED in $(seq 1 $NTEST);
    do
        python main.py --dyn_range 1000 --n 10  --resume models/$RUN/$SEED/ckpt \
        --vsa_conversion --vsa_selection --shared_rules --config $CONFIG --dataset $DATA --data_dir $DATA_DIR  \
        --batch_size 64 --num_workers 1 --num_rules $NRULES --num_terms 26 --seed $SEED --run_name $RUN \
        --exp_dir $EXP_DIR --partition _shuffle --mode test --evaluate-rule  --orientation-confounder $N_CONFOUNDERS  --entropy
    done
    python results.py --path $EXP_DIR/$RUN --seeds $NTEST
done
