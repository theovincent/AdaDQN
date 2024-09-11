#!/bin/bash

SHARED_ARGS="--first_seed 1 --last_seed 3 --n_parallel_seeds 2 --replay_buffer_capacity 1_000_000 --batch_size 32 \
    --update_horizon 1 --gamma 0.99 --horizon 27_000 --n_training_steps_per_epoch 250_000 --update_to_data 4 \
    --n_initial_samples 20_000 --cnn_n_layers_range 1 5 --cnn_n_channels_range 16 80 --cnn_kernel_size_range 2 10 \
    --cnn_stride_range 1 5 --mlp_n_layers_range 0 2 --mlp_n_neurons_range 25 1024 --learning_rate_range 6 3"

for TARGET_UPDATE_FREQUENCY in 1000 4000 8000
do
    for HP_UPDATE_FREQUENCY_FACTOR in 10 30 60
    do
        HP_UPDATE_FREQUENCY=$(( $HP_UPDATE_FREQUENCY_FACTOR * $TARGET_UPDATE_FREQUENCY ))
        for EXPLOITATION_TYPE in elitism truncation
        do
            for RESET_MODE in reset no_reset
            do
                if [[ $RESET_MODE = "reset" ]]
                then
                    RESET_FLAG="--reset_weights"
                else
                    RESET_FLAG=""
                fi
                launch_job/atari/cluster_adadqn.sh --experiment_name tuf_$TARGET_UPDATE_FREQUENCY\_huf_$HP_UPDATE_FREQUENCY\_$EXPLOITATION_TYPE\_$RESET_MODE\_Pong \
                    $SHARED_ARGS --target_update_freq $TARGET_UPDATE_FREQUENCY --n_epochs 10 --n_networks 5 \
                    --exploitation_type $EXPLOITATION_TYPE --epsilon_end 0.01 --epsilon_duration 250_000 \
                    --hp_update_frequency $HP_UPDATE_FREQUENCY $RESET_FLAG
            done
        done
        sleep 10m
    done
done

# for MIN_STEPS_EVALUATION in 400 1000 2000
# do
#     launch_job/atari/cluster_searldqn.sh --experiment_name min_steps_eval_$MIN_STEPS_EVALUATION\_elitism_no_reset \
#         $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type elitism --min_steps_evaluation $MIN_STEPS_EVALUATION \
#         --training_proportion 0.5

#     launch_job/atari/cluster_searldqn.sh --experiment_name min_steps_eval_$MIN_STEPS_EVALUATION\_truncation_no_reset \
#         $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type truncation --min_steps_evaluation $MIN_STEPS_EVALUATION \
#         --training_proportion 0.5
#     sleep 15 min
# done

# launch_job/atari/cluster_rsdqn.sh --experiment_name ne30 $SHARED_ARGS --n_epochs 300 --epsilon_end 0.01 \
#     --epsilon_duration 1_000 --hp_update_per_epoch 30

# launch_job/atari/cluster_dehbdqn.sh --experiment_name minne10_maxne50 $SHARED_ARGS --n_epochs 300 --epsilon_end 0.01 \
#     --epsilon_duration 1_000 --min_n_epochs_per_hp 10 --max_n_epochs_per_hp 50