#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

if [[ $GPU = true ]]
then
    submission_train="sbatch -J $EXPERIMENT_NAME\_$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 --gres=gpu:1 -p amd,amd2,amd3 \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$ALGO_NAME_%a.out \
    launch_job/$ENV_NAME/train_$ALGO_NAME.sh -e $EXPERIMENT_NAME $ARGS -g"
else
    submission_train="sbatch -J $EXPERIMENT_NAME\_$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=1500M --time=03:30:00 -p amd,amd2,amd3 \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$ALGO_NAME_%a.out \
    launch_job/$ENV_NAME/train_$ALGO_NAME.sh -e $EXPERIMENT_NAME $ARGS"
fi