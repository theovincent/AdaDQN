import os
import shutil
import subprocess
import unittest


class TestAtari(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_args = (
            "--seed 1 --disable_wandb --replay_buffer_capacity 100 --batch_size 3 --update_horizon 1 --gamma 0.99 "
            + "--horizon 10 --n_epochs 3 --n_training_steps_per_epoch 5 --update_to_data 3 "
            + "--target_update_frequency 3 --n_initial_samples 3 --cnn_n_layers_range 1 3 --cnn_n_channels_range 12 64 "
            + "--cnn_kernel_size_range 2 8 --cnn_stride_range 2 8 --mlp_n_layers_range 1 3 --mlp_n_neurons_range 25 200 "
            + "--activations celu elu tanh --losses huber l1 --optimizers adagrad nadam --learning_rate_range 4 2"
        )

    def run_core_test(self, algo_name, algo_args):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"../experiments/atari/exp_output/_test_{algo_name}_Pong"
        )
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            (
                f"python3 experiments/atari/{algo_name}.py --experiment_name _test_{algo_name}_Pong {self.base_args} {algo_args}"
            ).split(" ")
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        shutil.rmtree(save_path)

    def test_adadqn(self):
        self.run_core_test(
            "adadqn",
            "--n_networks 5 --exploitation_type elitism --epsilon_end 0.01 --epsilon_duration 4 --hp_update_frequency 3",
        )
        self.run_core_test(
            "adadqn",
            "--n_networks 5 --exploitation_type elitism --epsilon_end 0.01 --epsilon_duration 4 --hp_update_frequency 3 --reset_weights",
        )

    def test_searldqn(self):
        self.run_core_test(
            "searldqn",
            "--n_networks 5 --exploitation_type elitism --min_steps_evaluation 1",
        )
        self.run_core_test(
            "searldqn",
            "--n_networks 5 --exploitation_type elitism --min_steps_evaluation 1 --reset_weights",
        )

    def test_rsdqn(self):
        self.run_core_test("rsdqn", "--epsilon_end 0.01 --epsilon_duration 4 --hp_update_per_epoch 1")

    def test_dehbdqn(self):
        self.run_core_test(
            "dehbdqn", "--epsilon_end 0.01 --epsilon_duration 4 --min_n_epochs_per_hp 1 --max_n_epochs_per_hp 2"
        )
