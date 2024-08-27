#!/bin/bash

shared_args="--experiment_name adadqn_vs_baselines --first_seed 1 --last_seed 20 --n_parallel_seeds 1 \
    --replay_buffer_capacity 10_000 --batch_size 32 --update_horizon 1 --gamma 0.99 --horizon 1_000 \
    --n_training_steps_per_epoch 10_000 --update_to_data 1 --target_update_frequency 200 --n_initial_samples 1_000 \
    --n_layers_range 1 3 --n_neurons_range 25 200 --learning_rate_range 5 2 \ 

launch_job/lunar_lander/cluster_adadqn.sh --n_epochs 50 --n_networks 5 --exploitation_type elitism \
    --epsilon_end 0.01 --epsilon_duration 1_000 --hp_update_frequency 2_000 --epsilon_online_end 1 \
    --epsilon_online_duration 500_000

launch_job/lunar_lander/cluster_searldqn.sh --n_epochs 50 --n_networks 5 --exploitation_type elitism \
    --min_steps_evaluation 400 --training_proportion 0.5

launch_job/lunar_lander/cluster_rsdqn.sh --n_epochs 200 --epsilon_end 0.01 --epsilon_duration 1_000 \
    --hp_update_per_epoch 20

launch_job/lunar_lander/cluster_dehbdqn.sh --n_epochs 200 --epsilon_end 0.01 --epsilon_duration 1_000 \
    --min_n_epochs_per_hp 13 --max_n_epochs_per_hp 27