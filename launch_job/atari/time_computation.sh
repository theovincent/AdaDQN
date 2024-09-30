#!/bin/bash

if ! tmux has-session -t slimdqn; then
    tmux new-session -d -s slimdqn
    echo "Created new tmux session - slimdqn"
fi

[ -d experiments/atari/logs/_test_time_computation ] || mkdir -p experiments/atari/logs/_time_computation

tmux send-keys -t slimdqn "cd $(pwd)" ENTER
tmux send-keys -t slimdqn "source env_gpu/bin/activate" ENTER
tmux send-keys -t slimdqn "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9" ENTER

echo "Compute time"
tmux send-keys -t slimdqn "python3 tests/time_computation/time_algorithms.py >> experiments/atari/logs/_time_computation/time.out 2>&1" ENTER