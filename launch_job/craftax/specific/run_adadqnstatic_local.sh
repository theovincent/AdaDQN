#!/bin/bash


launch_job/craftax/launch_local_adadqnstatic.sh -e lr_5_4_3_fs_512_512_256_1024_512_256_act_relu_tanh "-frs 1 -lrs 1 -g \
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
    -nn 9 \
    -osl adam adam adam adam adam adam adam adam adam \
    -lrsl 1e-5 1e-4 1e-3 1e-5 1e-4 1e-3 1e-5 1e-4 1e-3 \
    -lsl l2 l2 l2 l2 l2 l2 l2 l2 l2 \
    -fsl 512,512,256 512,512,256 512,512,256 512,512,256 512,512,256 512,512,256 1024,512,256 1024,512,256 1024,512,256 \
    -asl relu,relu,relu relu,relu,relu relu,relu,relu tanh,tanh,tanh tanh,tanh,tanh tanh,tanh,tanh tanh,tanh,tanh tanh,tanh,tanh tanh,tanh,tanh \
    -eoe 0.01"

