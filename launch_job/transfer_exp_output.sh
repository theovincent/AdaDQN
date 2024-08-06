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


EXPERIMENT_NAME=o01_a1_more_ram
scp -r vincent@mn.ias.informatik.tu-darmstadt.de:/mnt/beegfs/home/vincent/SlimAdaQN/experiments/lunar_lander/exp_output/$EXPERIMENT_NAME experiments/lunar_lander/exp_output/
