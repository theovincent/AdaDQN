#!/bin/bash

for LEARNING_RATE in 1e-5 1e-4 1e-3
do
    for FEATURES in "512 512 256" "1024 512 256"
    do
        for ACTIVATIONS in "relu relu relu" "tanh tanh tanh"
        do
            launch_job/craftax/launch_cluster_dqn.sh -e lr_${LEARNING_RATE//"1e-"/}\_fs_${FEATURES// /_}\_act_${ACTIVATIONS::4} "-frs 1 -lrs 1 -nps 2 \
                -rb 200_000 \
                -bs 32 \
                -n 1 \
                -gamma 0.99 \
                -hor 10_000 \
                -utd 1 \
                -tuf 500 \
                -nis 1_000 \
                -eps_e 0.01 \
                -eps_dur 250_000 \
                -ne 100 \
                -spe 10_000 \
                -o adam \
                -lr $LEARNING_RATE \
                -l l2 \
                -fs $FEATURES \
                -as $ACTIVATIONS"
        done
    done
done