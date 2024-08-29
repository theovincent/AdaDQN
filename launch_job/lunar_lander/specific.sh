#!/bin/bash

SHARED_ARGS="--first_seed 1 --last_seed 20 --n_parallel_seeds 1 --replay_buffer_capacity 10_000 \
    --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 1_000 --n_training_steps_per_epoch 10_000 \
    --update_to_data 1 --target_update_frequency 200 --n_initial_samples 1_000 --n_layers_range 1 3 \
    --n_neurons_range 25 200 --learning_rate_range 5 2"

for HP_UPDATE_FREQUENCY in 2_000 10_000 15_000
do
    launch_job/lunar_lander/cluster_adadqn.sh --experiment_name hp_update_freq_$HP_UPDATE_FREQUENCY\_elitism_no_reset \
        $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type elitism --epsilon_end 0.01 \
        --epsilon_duration 1_000 --hp_update_frequency $HP_UPDATE_FREQUENCY

    launch_job/lunar_lander/cluster_adadqn.sh --experiment_name hp_update_freq_$HP_UPDATE_FREQUENCY\_truncation_no_reset \
        $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type truncation --epsilon_end 0.01 \
        --epsilon_duration 1_000 --hp_update_frequency $HP_UPDATE_FREQUENCY
    sleep 15 min
done

for MIN_STEPS_EVALUATION in 400 1000 2000
do
    launch_job/lunar_lander/cluster_searldqn.sh --experiment_name min_steps_eval_$MIN_STEPS_EVALUATION\_elitism_no_reset \
        $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type elitism --min_steps_evaluation $MIN_STEPS_EVALUATION \
        --training_proportion 0.5

    launch_job/lunar_lander/cluster_searldqn.sh --experiment_name min_steps_eval_$MIN_STEPS_EVALUATION\_truncation_no_reset \
        $SHARED_ARGS --n_epochs 50 --n_networks 5 --exploitation_type truncation --min_steps_evaluation $MIN_STEPS_EVALUATION \
        --training_proportion 0.5
    sleep 15 min
done

# launch_job/lunar_lander/cluster_rsdqn.sh --experiment_name ne30 $SHARED_ARGS --n_epochs 300 --epsilon_end 0.01 \
#     --epsilon_duration 1_000 --hp_update_per_epoch 30

# launch_job/lunar_lander/cluster_dehbdqn.sh --experiment_name minne10_maxne50 $SHARED_ARGS --n_epochs 300 --epsilon_end 0.01 \
#     --epsilon_duration 1_000 --min_n_epochs_per_hp 10 --max_n_epochs_per_hp 50