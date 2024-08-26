import jax
import numpy as np
import optax
from tqdm import tqdm

from experiments.base.utils import save_data
from slimdqn.networks.adadqn import AdaDQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import collect_single_sample


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: AdaDQN,
    env,
    rb: ReplayBuffer,
):
    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"])

    n_training_steps = 0
    env.reset()
    episode_returns_per_epoch = [[0]]
    episode_lengths_per_epoch = [[0]]
    cumulated_loss = 0

    logs = {}
    for idx_hp, hp_detail in enumerate(agent.hp_details):
        for k_, v_ in hp_detail.items():
            logs[f"hps/{idx_hp}_{k_}"] = v_
    p["wandb"].log({"n_training_steps": n_training_steps, **logs})

    for idx_epoch in tqdm(range(p["n_epochs"])):
        n_training_steps_epoch = 0
        has_reset = False

        while n_training_steps_epoch < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key = jax.random.split(key)
            reward, has_reset = collect_single_sample(
                exploration_key, env, agent, rb, p["horizon"], epsilon_schedule, n_training_steps
            )

            n_training_steps_epoch += 1
            n_training_steps += 1

            episode_returns_per_epoch[idx_epoch][-1] += reward
            episode_lengths_per_epoch[idx_epoch][-1] += 1
            if has_reset and n_training_steps_epoch < p["n_training_steps_per_epoch"]:
                episode_returns_per_epoch[idx_epoch].append(0)
                episode_lengths_per_epoch[idx_epoch].append(0)

            if n_training_steps > p["n_initial_samples"]:
                cumulated_loss += agent.update_online_params(n_training_steps, rb)
                target_updated, hp_changed = agent.update_target_params(n_training_steps)

                if target_updated:
                    logs = {
                        "n_training_steps": n_training_steps,
                        "loss": cumulated_loss,
                        "idx_compute_target": agent.idx_compute_target,
                        "idx_draw_action": agent.idx_draw_action,
                    }
                    if hp_changed:
                        for idx_hp in agent.indices_new_hps:
                            for k_, v_ in agent.hp_details[idx_hp].items():
                                logs[f"hps/{idx_hp}_{k_}"] = v_
                    p["wandb"].log(logs)

                    cumulated_loss = 0

        avg_return = np.mean(episode_returns_per_epoch[idx_epoch])
        avg_length_episode = np.mean(episode_lengths_per_epoch[idx_epoch])
        n_episodes = len(episode_lengths_per_epoch[idx_epoch])
        print(
            f"\nEpoch {idx_epoch}: Return {avg_return} averaged on {n_episodes} episodes.\n",
            flush=True,
        )
        p["wandb"].log(
            {
                "epoch": idx_epoch,
                "n_training_steps": n_training_steps,
                "avg_return": avg_return,
                "avg_length_episode": avg_length_episode,
            }
        )

        if idx_epoch < p["n_epochs"] - 1:
            episode_returns_per_epoch.append([0])
            episode_lengths_per_epoch.append([0])

        save_data(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())
