#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/logs/$EXPERIMENT_NAME/AdaDQN ] || mkdir -p experiments/lunar_lander/logs/$EXPERIMENT_NAME/AdaDQN

echo "launch train adadqn"

if [[ $GPU = true ]]
then
    submission_train_adadqn="sbatch -J $EXPERIMENT_NAME\_adadqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 --gres=gpu:1 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/AdaDQN/train_adadqn_%a.out \
    launch_job/lunar_lander/train_adadqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $HP_SEARCH_ARGS $ADADQN_ARGS -g"
else
    submission_train_adadqn="sbatch -J $EXPERIMENT_NAME\_adadqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=1500M --time=03:30:00 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/AdaDQN/train_adadqn_%a.out \
    launch_job/lunar_lander/train_adadqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $HP_SEARCH_ARGS $ADADQN_ARGS"
fi

$submission_train_adadqn
