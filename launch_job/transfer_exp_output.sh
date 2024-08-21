#!/bin/bash

# for BASE_EXPERIMENT_NAME in epsilon
# do
#     for GAME in BattleZone # BattleZone DoubleDunk NameThisGame
#     do
#         for HP in adam_small_medium_large
#         do
#             EXPERIMENT_NAME=$BASE_EXPERIMENT_NAME\_$HP\_$GAME
#             scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/atari/exp_output/$EXPERIMENT_NAME experiments/atari/exp_output/
#         done
#     done
# done


# for BASE_EXPERIMENT_NAME in epsilon
# do
#     for GAME in NameThisGame # BattleZone DoubleDunk NameThisGame
#     do
#         for HP in adam_small_medium_large
#         do
#             EXPERIMENT_NAME=$BASE_EXPERIMENT_NAME\_$HP\_$GAME
#             # scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/atari/exp_output/$EXPERIMENT_NAME experiments/atari/exp_output/
#             scp -P 449 -r theo@dfki-r-pc002.ai.tu-darmstadt.de:/home/theo/SlimAdaQN/experiments/atari/exp_output/$EXPERIMENT_NAME experiments/atari/exp_output/
#         done
#     done
# done


# EXPERIMENT_NAME=o01_a1_more_ram
# scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/lunar_lander/exp_output/$EXPERIMENT_NAME experiments/lunar_lander/exp_output/


EXPERIMENT_NAME=TUF_500_loss_1_2_fs_512_512_256_1024_512_256_act_relu_tanh
# scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/craftax/exp_output/$EXPERIMENT_NAME experiments/craftax/exp_output/
scp -P 449 -r theo@dfki-r-pc002.ai.tu-darmstadt.de:/home/theo/SlimAdaQN/experiments/craftax/exp_output/$EXPERIMENT_NAME experiments/craftax/exp_output/


for LOSS in l1 l2
do
    for FEATURES in 512_512_256 1024_512_256
    do
        for ACTIVATION in relu tanh
        do
            EXPERIMENT_NAME=TUF_500_loss_$LOSS\_fs_$FEATURES\_act_$ACTIVATION
            scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/craftax/exp_output/$EXPERIMENT_NAME experiments/craftax/exp_output/
        done
    done
done
