#!/bin/bash


for GAME in "NameThisGame" # "BattleZone" "DoubleDunk" "NameThisGame"
do
    launch_job/atari/launch_local_adadqnstatic.sh -e architecture_44_52_64_512__32_64_64_512__46_46_46_714_$GAME "-frs 1 -lrs 2 -g \
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
        -lrsl 5e-5 5e-5 5e-5 \
        -lsl l2 l2 l2 \
        -fsl 44,52,64,512 32,64,64,512 46,46,46,714 \
        -asl relu,relu,relu,relu relu,relu,relu,relu relu,relu,relu,relu \
        -eoe 0.01"
done

