import os
import sys

import jax

from experiments.base.individual_dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.lunar_lander import LunarLander
from slimdqn.networks.individual_dqn import DEHBDQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer

from slimdqn.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (
        os.path.abspath(__file__).split("/")[-2],
        os.path.abspath(__file__).split("/")[-1][:-3],
    )
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    p["hp_space"] = {
        "cnn_n_layers_range": p["cnn_n_layers_range"],
        "cnn_n_channels_range": p["cnn_n_channels_range"],
        "cnn_kernel_size_range": p["cnn_kernel_size_range"],
        "cnn_stride_range": p["cnn_stride_range"],
        "mlp_n_layers_range": p["mlp_n_layers_range"],
        "mlp_n_neurons_range": p["mlp_n_neurons_range"],
        "activations": [ACTIVATIONS[key] for key in p["activations"]],
        "losses": [LOSSES[key] for key in p["losses"]],
        "optimizers": [OPTIMIZERS[key] for key in p["optimizers"]],
        "learning_rate_range": p["learning_rate_range"],
        "reset_weights": True,
    }
    agent = DEHBDQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        hp_space=p["hp_space"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
        min_n_epochs_per_hp=p["min_n_epochs_per_hp"],
        max_n_epochs_per_hp=p["max_n_epochs_per_hp"],
    )

    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
