from typing import Dict
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
import numpy as np
from slimdqn.networks.hyperparameters.generators import HPGenerator
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import collect_single_episode

from slimdqn.sample_collection import IDX_RB


class SEARLDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        n_networks: int,
        hp_space,
        exploitation_type: str,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        hp_key, self.q_key, _ = jax.random.split(key, 3)
        self.n_networks = n_networks
        self.hp_generator = HPGenerator(hp_key, observation_dim, n_actions, hp_space, exploitation_type)

        self.hp_fns = [None] * self.n_networks
        self.params = [None] * self.n_networks
        self.optimizer_states = [None] * self.n_networks
        self.hp_details = [None] * self.n_networks

        for idx_hp in range(self.n_networks):
            self.q_key, hp_key = jax.random.split(self.q_key)
            self.hp_fns[idx_hp], self.params[idx_hp], self.optimizer_states[idx_hp], self.hp_details[idx_hp] = (
                self.hp_generator.sample(hp_key)
            )
            print(f"Starting HP: {self.hp_details[idx_hp]}", flush=True)

        if exploitation_type == "elitism":
            self.indices_new_hps = [None] * np.ceil((self.n_networks - 1) / 2).astype(int)
        elif exploitation_type == "truncation":
            self.indices_new_hps = [None] * np.ceil(self.n_networks * 0.2).astype(int)
        self.target_params = self.params.copy()
        self.losses = np.zeros(self.n_networks)

        self.idx_draw_action = 0

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch()
            self.losses += self.learn_on_batch(batch_samples)

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()
            # Reset the loss
            self.losses = np.zeros_like(self.losses)
            return True
        return False

    def learn_on_batch(self, batch_samples):
        losses = np.zeros(self.n_networks)

        for idx_hp in range(self.n_networks):
            value_next_states = self.hp_fns[idx_hp]["apply_fn"](
                self.target_params[idx_hp], batch_samples[IDX_RB["next_state"]]
            )
            targets = self.compute_target_from_values(value_next_states, batch_samples)

            loss, self.params[idx_hp], self.optimizer_states[idx_hp] = self.hp_fns[idx_hp]["update_and_loss_fn"](
                self.params[idx_hp], targets, batch_samples, self.optimizer_states[idx_hp]
            )

            losses[idx_hp] = loss

        return losses

    @partial(jax.jit, static_argnames="self")
    def compute_target_from_values(self, value_next_states, batch_samples):
        # computes the target value for single or a batch of samples
        return batch_samples[IDX_RB["reward"]] + (
            1 - batch_samples[IDX_RB["terminal"]]
        ) * self.gamma**self.update_horizon * jnp.max(value_next_states, axis=-1)

    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        return self.hp_fns[self.idx_draw_action]["best_action_fn"](params[self.idx_draw_action], state)

    def get_model(self) -> Dict:
        model = {}

        for idx_hp in range(self.n_networks):
            model[f"model_{idx_hp}"] = {
                "params": self.params[idx_hp],
                "hp_detail": self.hp_details[idx_hp],
            }

        return model

    def evaluate(self, env, rb, horizon, min_steps):
        returns = np.zeros(self.n_networks)
        n_steps = np.zeros(self.n_networks)

        for idx_hp in range(self.n_networks):
            self.idx_draw_action = idx_hp
            returns[idx_hp], n_steps[idx_hp] = collect_single_episode(env, self, rb, horizon, min_steps)

        return returns, n_steps

    def exploit_and_explore(self, returns):
        self.q_key, hp_key = jax.random.split(self.q_key)
        self.indices_new_hps, self.hp_fns, self.params, self.optimizer_states, self.hp_details = (
            self.hp_generator.exploit_and_explore(
                hp_key, returns, self.hp_fns, self.params, self.optimizer_states, self.hp_details
            )
        )

        for idx_hp in self.indices_new_hps:
            print(f"New HP: {self.hp_details[idx_hp]}", flush=True)

        # Force target update
        self.update_target_params(0)
