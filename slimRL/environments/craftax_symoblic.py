import numpy as np
import jax
from craftax.craftax_env import make_craftax_env_from_name


class CraftaxEnv:
    def __init__(self, key: jax.Array) -> None:
        self.key = key

        self.env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", auto_reset=False)
        self.env_step_fn = jax.jit(self.env.step)
        self.env_reset_fn = jax.jit(self.env.reset)

        self.observation_shape = self.env.observation_space(self.env.default_params).shape
        self.n_actions = self.env.action_space().n

    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state)

    def reset(self) -> None:
        self.key, reset_key = jax.random.split(self.key)
        self.state, self.env_state = self.env_reset_fn(reset_key)

        self.n_steps = 0

    def step(self, action):
        self.key, step_key = jax.random.split(self.key)

        self.n_steps += 1
        self.state, self.env_state, reward, absorbing, self.info = self.env_step_fn(
            step_key, self.env_state, action, self.env.default_params
        )

        return reward.item(), absorbing.item()
