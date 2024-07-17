#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN ] || mkdir -p experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN

echo "launch train rsdqn"

if [[ $GPU = true ]]
then
    submission_train_rsdqn="sbatch -J $EXPERIMENT_NAME\_rsdqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 --gres=gpu:1 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN/train_rsdqn_%a.out \
    launch_job/lunar_lander/train_rsdqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $HP_SEARCH_ARGS -g"
else
    submission_train_rsdqn="sbatch -J $EXPERIMENT_NAME\_rsdqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=1500M --time=03:30:00 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN/train_rsdqn_%a.out \
    launch_job/lunar_lander/train_rsdqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $HP_SEARCH_ARGS"
fi

$submission_train_rsdqn
