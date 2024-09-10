import os
import sys

import jax
import numpy as np

from experiments.base.pbt_dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.networks.searldqn import SEARLDQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer

from slimdqn.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name, algo_name = (
        os.path.abspath(__file__).split("/")[-2],
        os.path.abspath(__file__).split("/")[-1][:-3],
    )
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, _ = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    rb = ReplayBuffer(
        observation_shape=(env.state_height, env.state_width),
        replay_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
        stack_size=4,
        observation_dtype=np.uint8,
        terminal_dtype=np.uint8,
        action_dtype=np.int32,
        reward_dtype=np.float32,
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
    }
    agent = SEARLDQN(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        n_networks=p["n_networks"],
        hp_space=p["hp_space"],
        exploitation_type=p["exploitation_type"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )

    train(p, agent, env, rb)


if __name__ == "__main__":
    run()
