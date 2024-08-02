#!/bin/bash

for GAME in "BattleZone" "DoubleDunk" "NameThisGame"
do
    for HP in "1e-5" "5e-5" "1e-4"
    do
        launch_job/atari/launch_cluster_dqn.sh -e lr_$HP\_$GAME "-frs 1 -lrs 2 -nps 2 \
            -rb 1_000_000 \
            -bs 32 \
            -n 1 \
            -gamma 0.99 \
            -hor 27_000 \
            -utd 4 \
            -tuf 8_000 \
            -nis 20_000 \
            -eps_e 0.01 \
            -eps_dur 250_000 \
            -ne 40 \
            -spe 250_000 \
            -o adam \
            -lr $HP \
            -l l2 \
            -fs 32 64 64 512 \
            -as relu relu relu relu"


        launch_job/atari/launch_cluster_dqn.sh -e lr_$HP\_$GAME "-frs 5 -lrs 5 -nps 1 \
            -rb 1_000_000 \
            -bs 32 \
            -n 1 \
            -gamma 0.99 \
            -hor 27_000 \
            -utd 4 \
            -tuf 8_000 \
            -nis 20_000 \
            -eps_e 0.01 \
            -eps_dur 250_000 \
            -ne 40 \
            -spe 250_000 \
            -o adam \
            -lr $HP \
            -l l2 \
            -fs 32 64 64 512 \
            -as relu relu relu relu"
    done
done