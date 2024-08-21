#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if ! tmux has-session -t adadqn; then
    tmux new-session -d --seed adadqn
    echo "Created new tmux session - adadqn"
fi

if [[ $GPU = true ]]
then
    tmux send-keys -t adadqn "source env_gpu/bin/activate" ENTER
else
    tmux send-keys -t adadqn "source env_cpu/bin/activate" ENTER
fi

echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t adadqn\
    "python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t adadqn "wait" ENTER
