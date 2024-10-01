import json
import os
import shutil
import subprocess
import time

import numpy as np


def run_algorithm(algo_name, algo_args):
    print(f"\n\n\n--------------- Time {algo_name} ---------------", flush=True)
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"../../experiments/atari/exp_output/_time_{algo_name}_Pong"
    )
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    time_begin = time.time()
    returncode = subprocess.run(
        f"python3 experiments/atari/{algo_name}.py --experiment_name _time_{algo_name}_Pong {algo_args}".split(" ")
    ).returncode
    time_end = time.time()

    if returncode != 0:
        print(
            f"Training {algo_name} should not have raised an error. The training time has to be recomputed.", flush=True
        )
    else:
        print(f"{algo_name} trained in {np.around(time_end - time_begin)} seconds.", flush=True)

    shutil.rmtree(save_path)

    return time_end - time_begin if returncode == 0 else None


if __name__ == "__main__":
    base_args = (
        "--seed 1 --disable_wandb --replay_buffer_capacity 1_000_000 --batch_size 32 --update_horizon 1 --gamma 0.99 "
        + "--horizon 27_000 --target_update_freq 8000 --n_epochs 2 --n_training_steps_per_epoch 250_000 --update_to_data 4 "
        + "--n_initial_samples 20_000 --cnn_n_layers_range 1 3 --cnn_n_channels_range 16 64 --cnn_kernel_size_range 2 8 "
        + "--cnn_stride_range 2 5 --mlp_n_layers_range 0 2 --mlp_n_neurons_range 25 512 --learning_rate_range 6 3"
    )

    time_rsdqn = run_algorithm("rsdqn", base_args + " --epsilon_end 0.01 --epsilon_duration 1 --hp_update_per_epoch 30")
    # time_dehbdqn = run_algorithm(
    #     "dehbdqn",
    #     base_args + " --epsilon_end 0.01 --epsilon_duration 1 --min_n_epochs_per_hp 20 --max_n_epochs_per_hp 40",
    # )
    time_searldqn = run_algorithm(
        "searldqn",
        base_args + " --n_networks 5 --exploitation_type elitism --min_steps_evaluation 8000",
    )
    time_adadqn = run_algorithm(
        "adadqn",
        base_args
        + " --n_networks 5 --exploitation_type elitism --epsilon_end 0.01 --epsilon_duration 1 --hp_update_frequency 80000",
    )

    json.dump(
        {"rsdqn": time_rsdqn, "searldqn": time_searldqn, "adadqn": time_adadqn},
        open("tests/time_computation/time_algorithms.json", "w"),
        indent=4,
    )
