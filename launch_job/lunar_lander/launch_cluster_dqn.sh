#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN ] || mkdir -p experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN

echo "launch train dqn"

if [[ $GPU = true ]]
then
    submission_train_dqn="sbatch -J $EXPERIMENT_NAME\_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=750M --time=00:15:00 --gres=gpu:1 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN/train_dqn_%a.out \
    launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $DQN_ARGS -g"
else
    submission_train_dqn="sbatch -J $EXPERIMENT_NAME\_dqn --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=1 --mem-per-cpu=1Gb --time=01:00:00 -p amd,amd2,amd3 \
    --output=experiments/lunar_lander/logs/$EXPERIMENT_NAME/DQN/train_dqn_%a.out \
    launch_job/lunar_lander/train_dqn.sh -e $EXPERIMENT_NAME $BASE_ARGS $DQN_ARGS"
fi
    
$submission_train_dqn
