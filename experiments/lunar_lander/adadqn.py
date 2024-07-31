import os
import sys
import json
import jax
import numpy as np
from experiments.base.parser import adadqn_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.adadqn import AdaDQN
from experiments.base.dqn import train
from experiments.base.utils import prepare_logs

from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name = os.path.abspath(__file__).split(os.sep)[-2]
    p = adadqn_parser(env_name, argvs)

    prepare_logs(p)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        stack_size=1,
        observation_dtype=np.float32,
        terminal_dtype=np.uint8,
        action_dtype=np.int32,
        reward_dtype=np.float32,
    )
    agent = AdaDQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        n_networks=p["n_networks"],
        optimizers=[OPTIMIZERS[key] for key in p["optimizers"]],
        lr_range=p["lr_range"],
        losses=[LOSSES[key] for key in p["losses"]],
        n_layers_range=p["n_layers_range"],
        n_neurons_range=p["n_neurons_range"],
        activations=[ACTIVATIONS[key] for key in p["activations"]],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
        end_online_exp=p["end_online_exp"],
        duration_online_exp=p["n_epochs"] * p["n_training_steps_per_epoch"],
        optimizer_change_probability=p["optimizer_change_probability"],
        architecture_change_probability=p["architecture_change_probability"],
    )
    train(train_key, p, agent, env, rb)

    # Save extra data
    os.makedirs(os.path.join(p["save_path"], "indices_and_hyperparameters_details"), exist_ok=True)
    indices_and_hyperparameters_details_path = os.path.join(
        p["save_path"], f"indices_and_hyperparameters_details/{p['seed']}.json"
    )

    json.dump(
        {
            **jax.tree_map(
                int,
                {
                    "compute_target": agent.indices_compute_target,
                    "kicked_out": agent.indices_kicked_out,
                    "draw_action": agent.indices_draw_action,
                },
            ),
            "hyperparameters_details": agent.hyperparameters_details,
        },
        open(indices_and_hyperparameters_details_path, "w"),
        indent=4,
    )
