import jax
import numpy as np
import optax
from tqdm import tqdm

from experiments.base.utils import save_data
from slimdqn.networks.adadqn import AdaDQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import collect_single_episode


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: AdaDQN,
    env,
    rb: ReplayBuffer,
):
    n_training_steps = 0
    cumulated_loss = 0
    env.reset()

    logs = {}
    for idx_hp, hp_detail in enumerate(agent.hp_details):
        for k_, v_ in hp_detail.items():
            logs[f"hps/{idx_hp}_{k_}"] = v_
    p["wandb"].log({"n_training_steps": n_training_steps, **logs})

    while n_training_steps < p["n_epochs"] * p["n_training_steps_per_epoch"]:
        returns, n_steps = agent.evaluate(env, rb, p["horizon"], p["min_steps_evaluation"])

        n_training_steps += n_steps

        for _ in range(n_steps // 2):
            cumulated_loss += agent.update_online_params(rb)
            target_updated = agent.update_target_params(n_training_steps)

            if target_updated:
                p["wandb"].log(
                    {
                        "n_training_steps": n_training_steps,
                        "loss": cumulated_loss,
                    }
                )
                cumulated_loss = 0

        agent.exploit_and_explore(returns)

        logs = {}
        for idx_hp, hp_detail in enumerate(agent.hp_details):
            for k_, v_ in hp_detail.items():
                logs[f"hps/{idx_hp}_{k_}"] = v_
            logs[f"hps/{idx_hp}_return"] = returns[idx_hp]
        p["wandb"].log(
            {
                "n_training_steps": n_training_steps,
                "min_return": np.min(returns),
                "mean_return": np.mean(returns),
                "max_return": np.max(returns),
                **logs,
            }
        )

        print(
            f"\N training steps {n_training_steps}: Return {np.mean(returns)} +- {np.max(returns)} | {np.min(returns)}.\n",
            flush=True,
        )

        save_data(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())

        # Force target update
        agent.update_target_params(0)
