#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@ -frs dummy -lrs dummy

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi


if [[ $N_PARALLEL_SEEDS = 1 ]]
then
    $ENV_NAME\_$ALGO_NAME -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID $ARGS
else
    $ENV_NAME\_$ALGO_NAME -e $EXPERIMENT_NAME -s $((2 * SLURM_ARRAY_TASK_ID - 1)) $ARGS &> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$((2 * SLURM_ARRAY_TASK_ID - 1)).out & 
    $ENV_NAME\_$ALGO_NAME -e $EXPERIMENT_NAME -s $((2 * SLURM_ARRAY_TASK_ID)) $ARGS &> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$((2 * SLURM_ARRAY_TASK_ID)).out

    wait
fi