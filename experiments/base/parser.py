import os
import argparse
import time

from experiments import DISPLAY_NAME
from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def base_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for the experiment.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-rb",
        "--replay_capacity",
        help="Replay Buffer capacity.",
        type=int,
        default=200_000,
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
        help="Discounting factor gamma.",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "-hor",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-utd",
        "--update_to_data",
        help="No. of data points to collect per online Q-network update.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-tuf",
        "--target_update_frequency",
        help="Update period for target Q-network.",
        type=int,
        default=500,
    )

    parser.add_argument(
        "-n_init",
        "--n_initial_samples",
        help="No. of initial samples before training begins.",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "-eps_e",
        "--end_epsilon",
        help="Ending value of epsilon for linear schedule.",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "-eps_dur",
        "--duration_epsilon",
        help="Duration(number of steps) over which epsilon decays.",
        type=float,
        default=1_000,
    )

    parser.add_argument(
        "-ne",
        "--n_epochs",
        help="No. of epochs to train for.",
        type=int,
        default=50,
    )

    parser.add_argument(
        "-spe",
        "--n_training_steps_per_epoch",
        help="Max. no. of training steps per epoch.",
        type=int,
        default=10_000,
    )


def hyperparameter_search_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nlr",
        "--n_layers_range",
        nargs=2,
        help="Range of the number of layers.",
        type=int,
        default=[1, 3],
    )
    parser.add_argument(
        "-nnr",
        "--n_neurons_range",
        nargs=2,
        help="Range of the number of neurons per layers.",
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
        "-lrr",
        "--lr_range",
        nargs=2,
        help="Range of the optimizer's learning rate. It is sample in log space [10^low_range, 10^high_range].",
        type=int,
        default=[-6, -2],
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
        "-ls",
        "--losses",
        nargs="*",
        help="Losses.",
        type=str,
        choices=list(LOSSES.keys()),
        default=list(LOSSES.keys()),
    )


def adadqn_parser(env_name: str, argvs):
    algo_name = "adadqn"
    print(f"--- Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]} {time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    hyperparameter_search_parser(parser)
    parser.add_argument(
        "-nn",
        "--n_networks",
        help="No. of online Q-networks.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "-eoe",
        "--end_online_exp",
        help="End exploration espilon for sampling action.",
        type=float,
        default=0.01,
    )
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = env_name
    p["algo"] = algo_name
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )

    return p


def rsdqn_parser(env_name: str, argvs):
    algo_name = "rsdqn"
    print(f"--- Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]} {time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser(f"Train {DISPLAY_NAME[algo_name]} on {DISPLAY_NAME[env_name]}.")

    base_parser(parser)
    hyperparameter_search_parser(parser)
    parser.add_argument(
        "-sphp",
        "--n_training_step_per_hypeparameter",
        help="No. of training steps per hyperparameter update.",
        type=int,
        default=100_000,
    )
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = env_name
    p["algo"] = algo_name
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )

    return p
