#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

sbatch -J $EXPERIMENT_NAME\_$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=3 --mem-per-cpu=2G --time=15:00:00 --prefer="rtx3090|a5000" --gres=gpu:1 -p gpu \
--output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_%a.out \
launch_job/$ENV_NAME/train_$ALGO_NAME.sh --algo_name $ALGO_NAME --env_name $ENV_NAME -e $EXPERIMENT_NAME $ARGS -g
