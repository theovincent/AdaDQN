#!/bin/bash
source launch_job/parse_arguments.sh
parse_arguments $@ -frs dummy -lrs dummy

if [[ $GPU = true ]]
then
    source env_gpu/bin/activate
else
    source env_cpu/bin/activate
fi

lunar_lander_dqn -e $EXPERIMENT_NAME -s $SLURM_ARRAY_TASK_ID $BASE_ARGS $DQN_ARGS