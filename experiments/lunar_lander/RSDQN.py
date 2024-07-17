import os
import sys
import time
import argparse
import json
import jax
from experiments.base.parser import rsdqn_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.RSDQN import RSDQN
from experiments.base.DQN import train
from experiments.base.logger import prepare_logs

from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    print(f"---Lunar Lander__DQN__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
    rsdqn_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "Lunar Lander"
    p["algo"] = "RSDQN"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"exp_output/{p['experiment_name']}/{p['algo']}",
    )

    prepare_logs(p)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = RSDQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        n_layers_range=p["n_layers_range"],
        n_neurons_range=p["n_neurons_range"],
        activations=[ACTIVATIONS[key] for key in p["activations"]],
        lr_range=p["lr_range"],
        optimizers=[OPTIMIZERS[key] for key in p["optimizers"]],
        losses=[LOSSES[key] for key in p["losses"]],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )
    train(train_key, p, agent, env, rb)

    hyperparameters_details_path = os.path.join(p["save_path"], f"hyperparameters_details_{p['seed']}.json")
    json.dump(agent.hyperparameters_details, open(hyperparameters_details_path, "w"), indent=4)
