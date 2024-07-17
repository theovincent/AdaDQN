#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@

[ -d experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN ] || mkdir -p experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN

tmux has-session -t "slimRL" 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s slimRL
fi

if [[ $GPU = true ]]
then
    tmux send-keys -t slimRL "source env_gpu/bin/activate" ENTER
else
    tmux send-keys -t slimRL "source env_cpu/bin/activate" ENTER
fi

echo "launch train rsdqn local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimRL\
    "lunar_lander_rsdqn -e $EXPERIMENT_NAME -s $seed $BASE_ARGS $HP_SEARCH_ARGS >> experiments/lunar_lander/logs/$EXPERIMENT_NAME/RSDQN/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimRL "wait" ENTER