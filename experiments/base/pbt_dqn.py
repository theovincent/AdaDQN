import numpy as np
from tqdm import tqdm

from experiments.base.utils import save_data
from slimdqn.networks.searldqn import SEARLDQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer


def train(p: dict, agent: SEARLDQN, env, rb: ReplayBuffer):
    n_training_steps = 0
    env.reset()
    episode_returns_per_evaluation = []
    n_steps_per_evaluation = []

    logs = {"n_training_steps": n_training_steps}
    for idx_hp, hp_detail in enumerate(agent.hp_details):
        for k_, v_ in hp_detail.items():
            logs[f"hps/{idx_hp}_{k_}"] = v_
    p["wandb"].log(logs)

    pbar = tqdm(total=p["n_epochs"] * p["n_training_steps_per_epoch"])
    while n_training_steps < p["n_epochs"] * p["n_training_steps_per_epoch"]:
        returns, n_steps = agent.evaluate(env, rb, p["horizon"], p["min_steps_evaluation"])
        sum_n_steps = np.sum(n_steps)

        episode_returns_per_evaluation.append(list(returns))
        n_steps_per_evaluation.append(list(n_steps))
        n_training_steps += sum_n_steps
        pbar.update(sum_n_steps)

        for _ in range(int(sum_n_steps * p["training_proportion"])):
            agent.update_online_params(n_training_steps, rb)
            losses = agent.losses
            target_updated = agent.update_target_params(n_training_steps)

            if target_updated:
                logs = {"n_training_steps": n_training_steps}
                for idx_hp in range(p["n_networks"]):
                    logs[f"hps/{idx_hp}_loss"] = losses[idx_hp]
                p["wandb"].log(logs)

        agent.exploit_and_explore(returns)

        logs = {
            "n_training_steps": n_training_steps,
            "min_return": np.min(returns),
            "mean_return": np.mean(returns),
            "max_return": np.max(returns),
        }
        for idx_hp in range(p["n_networks"]):
            logs[f"hps/{idx_hp}_loss"] = losses[idx_hp]
            logs[f"hps/{idx_hp}_return"] = returns[idx_hp]
            logs[f"hps/{idx_hp}_n_steps"] = n_steps[idx_hp]
            if idx_hp in agent.indices_new_hps:
                for k_, v_ in agent.hp_details[idx_hp].items():
                    logs[f"hps/{idx_hp}_{k_}"] = v_
        p["wandb"].log(logs)

        print(
            f"\nTraining steps {n_training_steps}: Return {np.mean(returns)} +- {np.max(returns)} | {np.min(returns)}.\n",
            flush=True,
        )

        save_data(p, episode_returns_per_evaluation, n_steps_per_evaluation, agent.get_model())
