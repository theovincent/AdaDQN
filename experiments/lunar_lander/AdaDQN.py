import os
import sys
import time
import argparse
import json
import jax
from experiments.base.parser import adadqn_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.AdaDQN import AdaDQN
from experiments.base.DQN import train
from experiments.base.logger import prepare_logs


def run(argvs=sys.argv[1:]):
    print(f"---Lunar Lander__DQN__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
    adadqn_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "Lunar Lander"
    p["algo"] = "AdaDQN"
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
    agent = AdaDQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        n_networks=p["n_networks"],
        n_layers_range=p["n_layers_range"],
        n_neurons_range=p["n_neurons_range"],
        lr=p["lr"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        train_frequency=p["update_to_data"],
        target_update_frequency=p["target_update_period"],
        loss_type="huber",
        end_online_exp=p["end_online_exp"],
        duration_online_exp=p["n_epochs"] * p["n_training_steps_per_epoch"],
    )
    train(train_key, p, agent, env, rb)

    # Save selected networks for target computation and action selection
    compute_target_path = os.path.join(p["save_path"], f"indexes_compute_target_{p['seed']}.json")
    draw_action_path = os.path.join(p["save_path"], f"indexes_draw_action_{p['seed']}.json")

    json.dump(agent.indexes_compute_target, open(compute_target_path, "w"))
    json.dump(agent.indexes_draw_action, open(draw_action_path, "w"))
