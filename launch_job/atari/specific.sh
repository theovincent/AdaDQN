#!/bin/bash

SHARED_ARGS="--replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 27_000 \
    --n_training_steps_per_epoch 250_000 --update_to_data 4 --n_initial_samples 20_000 --cnn_n_layers_range 1 3 \
    --cnn_n_channels_range 16 64 --cnn_kernel_size_range 2 8 --cnn_stride_range 2 5 --mlp_n_layers_range 0 2 \
    --mlp_n_neurons_range 25 512 --learning_rate_range 6 3"

# -------------- AdaDQN --------------
# for TARGET_UPDATE_FREQUENCY in 4000 8000
# do
#     for HP_UPDATE_FREQUENCY in 40000 80000 
#     do
#         for EXPLOITATION_TYPE in elitism # elitism truncation
#         do
#             for RESET_MODE in no_reset # reset no_reset
#             do
#                 if [[ $RESET_MODE = "reset" ]]
#                 then
#                     RESET_FLAG="--reset_weights"
#                 else
#                     RESET_FLAG=""
#                 fi
#                 ADADQN_ARGS="--experiment_name long_tuf_${TARGET_UPDATE_FREQUENCY}_huf_${HP_UPDATE_FREQUENCY}_${EXPLOITATION_TYPE}_${RESET_MODE}_Pong \
#                     $SHARED_ARGS --target_update_freq $TARGET_UPDATE_FREQUENCY --n_epochs 40 --n_networks 5 \
#                     --exploitation_type $EXPLOITATION_TYPE --epsilon_end 0.01 --epsilon_duration 250_000 \
#                     --hp_update_frequency $HP_UPDATE_FREQUENCY $RESET_FLAG"

#                 launch_job/atari/cluster_adadqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $ADADQN_ARGS
#                 launch_job/atari/cluster_adadqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $ADADQN_ARGS
#             done
#         done
#     done
# done


# -------------- SEARL --------------
# for TARGET_UPDATE_FREQUENCY in 4000 8000
# do
#     for MIN_STEPS_EVALUATION in 8000 16000
#     do
#         for EXPLOITATION_TYPE in elitism truncation
#         do
#             for RESET_MODE in reset no_reset
#             do
#                 if [[ $RESET_MODE = "reset" ]]
#                 then
#                     RESET_FLAG="--reset_weights"
#                 else
#                     RESET_FLAG=""
#                 fi
#                 SEARLDQN_ARGS="--experiment_name tuf_${TARGET_UPDATE_FREQUENCY}_mse_${MIN_STEPS_EVALUATION}_tp_1_${EXPLOITATION_TYPE}_${RESET_MODE}_half_gradient_Pong \
#                     $SHARED_ARGS --target_update_freq $TARGET_UPDATE_FREQUENCY --n_epochs 10 --n_networks 5 \
#                     --exploitation_type $EXPLOITATION_TYPE --min_steps_evaluation $MIN_STEPS_EVALUATION \
#                     --training_proportion 0.5 $RESET_FLAG"

#                 launch_job/atari/cluster_searldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SEARLDQN_ARGS
#                 launch_job/atari/cluster_searldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SEARLDQN_ARGS
#             done
#         done
#     done
# done

# -------------- RSDQN --------------
RSDQN_ARGS="--experiment_name hpupe_30_RoadRunner $SHARED_ARGS --target_update_freq 8000 --n_epochs 180 \
    --epsilon_end 0.01 --epsilon_duration 250_000 --hp_update_per_epoch 30"
# launch_job/atari/local_rsdqn.sh --first_seed 1 --last_seed 2 $RSDQN_ARGS
launch_job/atari/local_rsdqn.sh --first_seed 5 --last_seed 5 $RSDQN_ARGS
# launch_job/atari/cluster_rsdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $RSDQN_ARGS
# launch_job/atari/cluster_rsdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $RSDQN_ARGS

# -------------- DEHB --------------
# DEHBDQN_ARGS="--experiment_name minnephp_5_maxnephp_15_Pong $SHARED_ARGS --target_update_freq 8000 --n_epochs 100 \
#     --epsilon_end 0.01 --epsilon_duration 250_000 --min_n_epochs_per_hp 5 --max_n_epochs_per_hp 15"
# launch_job/atari/local_dehbdqn.sh --first_seed 1 --last_seed 2 $DEHBDQN_ARGS
# launch_job/atari/local_dehbdqn.sh --first_seed 3 --last_seed 5 $DEHBDQN_ARGS
# launch_job/atari/cluster_dehbdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $DEHBDQN_ARGS
# launch_job/atari/cluster_dehbdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $DEHBDQN_ARGS