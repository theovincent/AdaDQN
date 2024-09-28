#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@ --first_seed dummy --last_seed dummy
FIRST_SEED=$((N_PARALLEL_SEEDS * (SLURM_ARRAY_TASK_ID - 1) + 1)) 
LAST_SEED=$((N_PARALLEL_SEEDS * SLURM_ARRAY_TASK_ID))

source env_gpu/bin/activate

if [[ $N_PARALLEL_SEEDS = 1 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.89
elif [[ $N_PARALLEL_SEEDS = 2 ]]
then
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.42
else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.275
fi

for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS &> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$seed.out & 
done
wait