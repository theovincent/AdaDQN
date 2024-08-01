#!/bin/bash

for GAME in "BattleZone" "DoubleDunk" "NameThisGame"
do
    launch_job/atari/launch_cluster_adadqnstatic.sh -e lr_1e-5_5e-5_1e-4_$GAME "-frs 1 -lrs 1 \
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
        -nn 3 \
        -osl adam adam adam \
        -lrsl 1e-5 5e-5 1e-4 \
        -lsl l2 l2 l2 \
        -fsl 32,64,64,512 32,64,64,512 32,64,64,512 \
        -asl relu,relu,relu,relu relu,relu,relu,relu relu,relu,relu,relu \
        -eoe 0.01"
done