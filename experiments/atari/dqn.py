import os
import sys
import jax
import numpy as np
from experiments.base.parser import dqn_parser
from slimRL.environments.atari import AtariEnv
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.dqn import DQN
from experiments.base.dqn import train
from experiments.base.utils import prepare_logs

from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name = os.path.abspath(__file__).split(os.sep)[-2]
    p = dqn_parser(env_name, argvs)

    prepare_logs(p)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    rb = ReplayBuffer(
        observation_shape=(env.state_height, env.state_width),
        replay_capacity=p["replay_capacity"],
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
    agent = DQN(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        optimizer=OPTIMIZERS[p["optimizer"]],
        learning_rate=p["learning_rate"],
        loss=LOSSES[p["loss"]],
        features=p["features"],
        activations=[ACTIVATIONS[key] for key in p["activations"]],
        cnn=True,
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )
    train(train_key, p, agent, env, rb)
