import argparse
from functools import wraps
from typing import Callable, List

from slimdqn.networks import ACTIVATIONS, LOSSES, OPTIMIZERS


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="Number of epochs to perform.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-ntspe",
        "--n_training_steps_per_epoch",
        help="Number of training steps per epoch.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="Number of data points to collect per online Q-network update.",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-tuf",
        "--target_update_frequency",
        help="Number of training steps before updating the target Q-network.",
        type=int,
        default=200,
    )
    parser.add_argument(
        "-nis",
        "--n_initial_samples",
        help="Number of initial samples before the training starts.",
        type=int,
        default=1_000,
    )
    # HP search space
    parser.add_argument(
        "-cnlr",
        "--cnn_n_layers_range",
        nargs=2,
        help="Range of the number of layers of the CNNs.",
        type=int,
        default=[0, 0],
    )
    parser.add_argument(
        "-cncr",
        "--cnn_n_channels_range",
        nargs=2,
        help="Range of the number of channels per layer of the CNNs.",
        type=int,
        default=[0, 0],
    )
    parser.add_argument(
        "-cksr",
        "--cnn_kernel_size_range",
        nargs=2,
        help="Range of the kernel size per layer of the CNNs.",
        type=int,
        default=[0, 0],
    )
    parser.add_argument(
        "-csr",
        "--cnn_stride_range",
        nargs=2,
        help="Range of the stride per layer of the CNNs.",
        type=int,
        default=[0, 0],
    )
    parser.add_argument(
        "-mnlr",
        "--mlp_n_layers_range",
        nargs=2,
        help="Range of the number of layers of the MLP.",
        type=int,
        default=[1, 3],
    )
    parser.add_argument(
        "-mnnr",
        "--mlp_n_neurons_range",
        nargs=2,
        help="Range of the number of neurons per layer of the MLP.",
        type=int,
        default=[50, 200],
    )
    parser.add_argument(
        "-as",
        "--activations",
        nargs="*",
        help="Activation functions.",
        type=str,
        choices=list(ACTIVATIONS.keys()),
        default=list(ACTIVATIONS.keys()),
    )
    parser.add_argument(
        "-ls",
        "--losses",
        nargs="*",
        help="Losses.",
        type=str,
        choices=list(LOSSES.keys()),
        default=list(LOSSES.keys()),
    )
    parser.add_argument(
        "-os",
        "--optimizers",
        nargs="*",
        help="Optimizers.",
        type=str,
        choices=list(OPTIMIZERS.keys()),
        default=list(OPTIMIZERS.keys()),
    )
    parser.add_argument(
        "-lrr",
        "--learning_rate_range",
        nargs=2,
        help="Range of the learning rate. It is sample in log space [10^-low_range, 10^-high_range].",
        type=int,
        default=[6, 2],
    )


def add_parallel_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nn",
        "--n_networks",
        help="Number of networks trained in parralel.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-et",
        "--exploitation_type",
        help="Type of exploitation in the space of hyperparameters.",
        type=str,
        choices=["elitism", "truncation"],
        default="elitism",
    )
    parser.add_argument(
        "-r",
        "--reset_weights",
        help="Reset the network weights for each new generation.",
        default=False,
        action="store_true",
    )


def add_epsilon_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ee",
        "--epsilon_end",
        help="Ending value for the linear decaying epsilon used for exploration.",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-ed",
        "--epsilon_duration",
        help="Duration of epsilon's linear decay used for exploration.",
        type=float,
        default=1_000,
    )


@output_added_arguments
def add_adadqn_arguments(parser: argparse.ArgumentParser):
    add_parallel_training_arguments(parser)
    add_epsilon_arguments(parser)
    parser.add_argument(
        "-huf",
        "--hp_update_frequency",
        help="Number of training steps before updating the hyperparameter.",
        type=int,
        default=10_000,
    )


@output_added_arguments
def add_searldqn_arguments(parser: argparse.ArgumentParser):
    add_parallel_training_arguments(parser)
    parser.add_argument(
        "-mse",
        "--min_steps_evaluation",
        help="Minimum number of steps to perform to evaluate an agent.",
        type=int,
        default=400,
    )


@output_added_arguments
def add_rsdqn_arguments(parser: argparse.ArgumentParser):
    add_epsilon_arguments(parser)
    parser.add_argument(
        "-hupe",
        "--hp_update_per_epoch",
        help="Number of epochs before updating the hyperparameter.",
        type=int,
        default=20,
    )


@output_added_arguments
def add_dehbdqn_arguments(parser: argparse.ArgumentParser):
    add_epsilon_arguments(parser)
    parser.add_argument(
        "-mimneph",
        "--min_n_epochs_per_hp",
        help="Minimum number of epochs before updating the hyperparameter.",
        type=int,
        default=13,
    )
    parser.add_argument(
        "-maxneph",
        "--max_n_epochs_per_hp",
        help="Maximum number of epochs before updating the hyperparameter.",
        type=int,
        default=27,
    )
