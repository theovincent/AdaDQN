from tqdm import tqdm
import jax
import optax
from slimRL.networks.DQN import DQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import collect_single_sample
from experiments.base.logger import save_logs


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    env,
    rb: ReplayBuffer,
):
    epsilon_schedule = optax.linear_schedule(1.0, p["end_epsilon"], p["duration_epsilon"])

    n_training_steps = 0
    key, reset_key = jax.random.split(key)
    env.reset(key=reset_key)
    episode_returns_per_epoch = [[0]]
    episode_lengths_per_epoch = [[0]]

    for idx_epoch in tqdm(range(p["n_epochs"])):
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key, sample_batch_key = jax.random.split(key, 3)
            reward, has_reset = collect_single_sample(
                exploration_key, env, agent, rb, p, epsilon_schedule, n_training_steps
            )

            episode_returns_per_epoch[idx_epoch][-1] += reward
            episode_lengths_per_epoch[idx_epoch][-1] += 1
            if has_reset:
                episode_returns_per_epoch[idx_epoch].append(0)
                episode_lengths_per_epoch[idx_epoch].append(0)

            if n_training_steps > p["n_initial_samples"]:
                agent.update_online_params(n_training_steps, sample_batch_key, p["batch_size"], rb)
                agent.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        print(
            f"Epoch: {idx_epoch}"
            + f" Avg. return = {sum(episode_returns_per_epoch[idx_epoch])/len(episode_lengths_per_epoch[idx_epoch])}"
            + f" Num episodes = {len(episode_lengths_per_epoch[idx_epoch])}",
            flush=True,
        )

        episode_returns_per_epoch.append([0])
        episode_lengths_per_epoch.append([0])

    save_logs(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())
