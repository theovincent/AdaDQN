import os
import sys
import jax
from experiments.base.parser import dqn_parser
from slimRL.environments.lunar_lander import LunarLander
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

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = DQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        optimizer=OPTIMIZERS[p["optimizer"]],
        learning_rate=p["lr"],
        loss=LOSSES[p["loss"]],
        hidden_layers=p["hidden_layers"],
        activations=[ACTIVATIONS[key] for key in p["activations"]],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )
    train(train_key, p, agent, env, rb)
