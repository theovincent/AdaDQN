import jax
import jax.numpy as jnp
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    epsilon_schedule,
    n_training_steps: int,
):
    key, epsilon_key = jax.random.split(key)

    if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
        key, sample_key = jax.random.split(key)
        action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    else:
        action = agent.best_action(agent.params, env.state).item()

    obs = env.state.copy()
    _, reward, termination = env.step(action)
    truncation = env.n_steps == p["horizon"]
    rb.add(obs, action, reward, termination, truncation)

    has_reset = termination or truncation
    if has_reset:
        env.reset(key=key)

    return reward, has_reset
