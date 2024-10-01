from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax._src.random import PRNGKey as PRNGKey

from slimdqn.networks.hyperparameters.generators import HPGenerator
from slimdqn.sample_collection import IDX_RB
from slimdqn.sample_collection.replay_buffer import ReplayBuffer


class IndividualDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        hp_space,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        hp_key, self.q_key, _ = jax.random.split(key, 3)
        self.hp_generator = HPGenerator(hp_key, observation_dim, n_actions, hp_space, None)

        self.loss = 0

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch()
            self.loss += self.learn_on_batch(batch_samples)

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()
            # Reset the loss
            self.loss = 0
            return True
        return False

    def learn_on_batch(self, batch_samples):
        value_next_states = self.hp_fn["apply_fn"](self.target_params, batch_samples[IDX_RB["next_state"]])
        targets = self.compute_target_from_values(value_next_states, batch_samples)
        loss, self.params, self.optimizer_state = self.hp_fn["update_and_loss_fn"](
            self.params, targets, batch_samples, self.optimizer_state
        )

        return loss

    @partial(jax.jit, static_argnames="self")
    def compute_target_from_values(self, value_next_states, batch_samples):
        # computes the target value for single or a batch of samples
        return batch_samples[IDX_RB["reward"]] + (
            1 - batch_samples[IDX_RB["terminal"]]
        ) * self.gamma**self.update_horizon * jnp.max(value_next_states, axis=-1)

    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        return self.hp_fn["best_action_fn"](params, state)

    def get_model(self) -> Dict:
        return {"params": self.params, "hp_detail": self.hp_detail}


class RSDQN(IndividualDQN):
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        hp_space,
        hp_update_per_epoch: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        super().__init__(
            key, observation_dim, n_actions, hp_space, gamma, update_horizon, update_to_data, target_update_frequency
        )
        self.q_key, hp_key = jax.random.split(self.q_key)
        self.hp_fn, self.params, self.optimizer_state, self.hp_detail = self.hp_generator.sample(hp_key)
        self.target_params = self.params.copy()
        print(f"Starting HP: {self.hp_detail}", flush=True)

        self.hp_update_per_epoch = hp_update_per_epoch

    def update_hp(self, idx_epoch, avg_return):
        if idx_epoch % self.hp_update_per_epoch == 0:
            self.q_key, hp_key = jax.random.split(self.q_key)
            self.hp_fn, self.params, self.optimizer_state, self.hp_detail = self.hp_generator.sample(hp_key)
            print(f"New HP: {self.hp_detail}", flush=True)

            self.target_params = self.params.copy()

            return True
        return False


class DEHBDQN(IndividualDQN):
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        hp_space,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        min_n_epochs_per_hp: int,
        max_n_epochs_per_hp: int,
    ):
        super().__init__(
            key, observation_dim, n_actions, hp_space, gamma, update_horizon, update_to_data, target_update_frequency
        )
        self.hp_generator.dehb_init(min_n_epochs_per_hp, max_n_epochs_per_hp)

        self.q_key, hp_key = jax.random.split(self.q_key)
        self.hp_fn, self.params, self.optimizer_state, self.hp_detail, self.hp_update_per_epoch = (
            self.hp_generator.dehb_sample(hp_key, None)
        )
        self.target_params = self.params.copy()
        print(f"Starting HP: {self.hp_detail}", flush=True)

    def update_hp(self, idx_epoch, avg_return):
        if idx_epoch % self.hp_update_per_epoch == 0:
            self.q_key, hp_key = jax.random.split(self.q_key)
            self.hp_fn, self.params, self.optimizer_state, self.hp_detail, self.hp_update_per_epoch = (
                self.hp_generator.dehb_sample(hp_key, avg_return)
            )
            print(f"New HP: {self.hp_detail}", flush=True)

            self.target_params = self.params.copy()

            return True
        return False
