import os
import sys
import json
import jax
import numpy as np
from experiments.base.parser import adadqnstatic_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.adadqnstatic import AdaDQNStatic
from experiments.base.dqn import train
from experiments.base.utils import prepare_logs

from slimRL.networks import ACTIVATIONS, OPTIMIZERS, LOSSES


def run(argvs=sys.argv[1:]):
    env_name = os.path.abspath(__file__).split(os.sep)[-2]
    p = adadqnstatic_parser(env_name, argvs)

    prepare_logs(p)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: x,
        stack_size=1,
        observation_dtype=np.float32,
        terminal_dtype=np.uint8,
        action_dtype=np.int32,
        reward_dtype=np.float32,
    )
    agent = AdaDQNStatic(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        n_networks=p["n_networks"],
        optimizers_list=[OPTIMIZERS[key] for key in p["optimizers_list"]],
        learning_rates_list=p["learning_rates_list"],
        losses_list=[LOSSES[key] for key in p["losses_list"]],
        features_list=p["features_list"],
        activations_list=[[ACTIVATIONS[key] for key in list_key] for list_key in p["activations_list"]],
        cnn=False,
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
        end_online_exp=p["end_online_exp"],
        duration_online_exp=p["n_epochs"] * p["n_training_steps_per_epoch"],
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
                int, {"compute_target": agent.indices_compute_target, "draw_action": agent.indices_draw_action}
            )
        },
        open(indices_and_hyperparameters_details_path, "w"),
        indent=4,
    )
