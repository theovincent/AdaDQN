import jax
import jax.numpy as jnp

from slimdqn.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    horizon: int,
    epsilon_schedule,
    n_training_steps: int,
):
    key, epsilon_key = jax.random.split(key)

    if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
        key, sample_key = jax.random.split(key)
        action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    else:
        action = agent.best_action(agent.params, env.state).item()

    obs = env.observation
    reward, absorbing = env.step(action)

    episode_end = absorbing or env.n_steps >= horizon
    rb.add(obs, action, reward, absorbing, episode_end=episode_end)

    if episode_end:
        env.reset()

    return reward, episode_end


def collect_single_episode(env, agent, rb: ReplayBuffer, horizon: int, min_steps: int):
    episode_end = False
    returns = 0
    n_episodes = 0
    n_steps = 0

    while not episode_end or n_steps < min_steps:
        action = agent.best_action(agent.params, env.state).item()
        obs = env.observation
        reward, absorbing = env.step(action)

        episode_end = absorbing or env.n_steps >= horizon
        rb.add(obs, action, reward, absorbing, episode_end=episode_end)

        returns += reward
        n_steps += 1

        if episode_end:
            env.reset()
            n_episodes += 1

    return returns / n_episodes, n_steps
