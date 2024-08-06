#!/bin/bash

launch_job/atari/launch_local_dqn.sh -e lr_5e-5_BattleZone "-frs 1 -lrs 1 -g \
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
    -lr 5e-5 \
    -l l2 \
    -fs 32 64 64 512 \
    -as relu relu relu relu"
