import os
import sys
import json
import jax
from experiments.base.parser import rsdqn_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.rsdqn import RSDQN
from experiments.base.dqn import train
from experiments.base.utils import prepare_logs

from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name = os.path.abspath(__file__).split(os.sep)[-2]
    p = rsdqn_parser(env_name, argvs)

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
        n_training_step_per_hypeparameter=p["n_training_step_per_hypeparameter"],
    )
    train(train_key, p, agent, env, rb)

    # Save extra data
    os.makedirs(os.path.join(p["save_path"], f"hyperparameters_details"), exist_ok=True)
    hyperparameters_details_path = os.path.join(p["save_path"], f"hyperparameters_details/{p['seed']}.json")

    json.dump(agent.hyperparameters_details, open(hyperparameters_details_path, "w"), indent=4)
