#!/bin/bash

SHARED_ARGS="--replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 27_000 \
    --n_training_steps_per_epoch 250_000 --update_to_data 4 --n_initial_samples 20_000 --cnn_n_layers_range 1 5 \
    --cnn_n_channels_range 16 80 --cnn_kernel_size_range 2 10 --cnn_stride_range 1 5 --mlp_n_layers_range 0 2 \
    --mlp_n_neurons_range 25 1024 --learning_rate_range 6 3"

# for TARGET_UPDATE_FREQUENCY in 1000 4000 8000
# do
#     for HP_UPDATE_FREQUENCY_FACTOR in 10 20 30
#     do
#         HP_UPDATE_FREQUENCY=$(( $HP_UPDATE_FREQUENCY_FACTOR * $TARGET_UPDATE_FREQUENCY ))
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
#                 ADADQN_ARGS="--experiment_name tuf_$TARGET_UPDATE_FREQUENCY\_huf_$HP_UPDATE_FREQUENCY\_$EXPLOITATION_TYPE\_$RESET_MODE\_Pong \
#                     $SHARED_ARGS --target_update_freq $TARGET_UPDATE_FREQUENCY --n_epochs 10 --n_networks 5 \
#                     --exploitation_type $EXPLOITATION_TYPE --epsilon_end 0.01 --epsilon_duration 250_000 \
#                     --hp_update_frequency $HP_UPDATE_FREQUENCY $RESET_FLAG"

#                 launch_job/atari/cluster_adadqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $ADADQN_ARGS
#                 launch_job/atari/cluster_adadqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $ADADQN_ARGS
#             done
#         done
#         sleep 10m
#     done
# done


# for TARGET_UPDATE_FREQUENCY in 1000 4000 8000
# do
#     for MIN_STEPS_EVALUATION in 200 1000 5000
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
#                 SEARLDQN_ARGS="--experiment_name tuf_$TARGET_UPDATE_FREQUENCY\_mse_$MIN_STEPS_EVALUATION\_tp_1_$EXPLOITATION_TYPE\_$RESET_MODE\_Pong \
#                     $SHARED_ARGS --target_update_freq $TARGET_UPDATE_FREQUENCY --n_epochs 10 --n_networks 5 \
#                     --exploitation_type $EXPLOITATION_TYPE --min_steps_evaluation $MIN_STEPS_EVALUATION \
#                     --training_proportion 1 $RESET_FLAG"

#                 launch_job/atari/cluster_searldqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $SEARLDQN_ARGS
#                 launch_job/atari/cluster_searldqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $SEARLDQN_ARGS
#             done
#         done
#         sleep 10m
#     done
# done

RSDQN_ARGS="--experiment_name hpupe_10 $SHARED_ARGS --n_epochs 100 --epsilon_end 0.01 --epsilon_duration 250_000 \
    --hp_update_per_epoch 10"
launch_job/atari/local_rsdqn.sh --first_seed 1 --last_seed 2 $RSDQN_ARGS
launch_job/atari/local_rsdqn.sh --first_seed 3 --last_seed 5 $RSDQN_ARGS
# launch_job/atari/cluster_rsdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $RSDQN_ARGS
# launch_job/atari/cluster_rsdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $RSDQN_ARGS

# DEHBDQN_ARGS="--experiment_name minnephp_5_maxnephp_15 $SHARED_ARGS --n_epochs 100 --epsilon_end 0.01 --epsilon_duration 250_000 \
#     --min_n_epochs_per_hp 5 --max_n_epochs_per_hp 15"
# launch_job/atari/local_dehbdqn.sh --first_seed 1 --last_seed 2 $DEHBDQN_ARGS
# launch_job/atari/local_dehbdqn.sh --first_seed 3 --last_seed 5 $DEHBDQN_ARGS
# launch_job/atari/cluster_dehbdqn.sh --first_seed 1 --last_seed 2 --n_parallel_seeds 2 $DEHBDQN_ARGS
# launch_job/atari/cluster_dehbdqn.sh --first_seed 5 --last_seed 5 --n_parallel_seeds 1 $DEHBDQN_ARGS