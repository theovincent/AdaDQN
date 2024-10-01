from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from slimdqn.networks.hyperparameters.generators import HPGenerator
from slimdqn.sample_collection import IDX_RB
from slimdqn.sample_collection.replay_buffer import ReplayBuffer


class AdaDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions: int,
        n_networks: int,
        hp_space,
        exploitation_type: str,
        hp_update_frequency: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        hp_key, self.q_key, self.action_key = jax.random.split(key, 3)
        self.n_networks = n_networks
        self.hp_generator = HPGenerator(hp_key, observation_dim, n_actions, hp_space, exploitation_type)
        self.hp_update_frequency = hp_update_frequency

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

        self.target_params = self.params[0].copy()
        self.idx_compute_target = 0
        self.losses = np.zeros(self.n_networks)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        assert (
            hp_update_frequency % target_update_frequency == 0
        ), f"hp_update_frequency ({hp_update_frequency}) should be a multiple of target_update_frequency ({target_update_frequency})."

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch()
            self.losses += self.learn_on_batch(batch_samples)

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            change_hp = step % self.hp_update_frequency == 0
            if change_hp:
                self.q_key, hp_key = jax.random.split(self.q_key)
                # for exploit_and_explore the higher the metric is the better
                # this is why -self.losses is given as input
                self.indices_new_hps, self.hp_fns, self.params, self.optimizer_states, self.hp_details = (
                    self.hp_generator.exploit_and_explore(
                        hp_key, -self.losses, self.hp_fns, self.params, self.optimizer_states, self.hp_details
                    )
                )

                for idx_hp in self.indices_new_hps:
                    print(f"New HP: {self.hp_details[idx_hp]}", flush=True)

            # Define new target | ignore the nans, if all nans take the last network (idx_compute_target = -1)
            self.idx_compute_target = jnp.nanargmin(self.losses)
            self.target_params = self.params[self.idx_compute_target].copy()

            # Reset the loss
            self.losses = np.zeros_like(self.losses)

            return True, change_hp
        return False, False

    def learn_on_batch(self, batch_samples):
        losses = np.zeros(self.n_networks)

        value_next_states = self.hp_fns[self.idx_compute_target]["apply_fn"](
            self.target_params, batch_samples[IDX_RB["next_state"]]
        )
        targets = self.compute_target_from_values(value_next_states, batch_samples)

        for idx_hp in range(self.n_networks):
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
        # computes the best action for a single state
        self.action_key, self.idx_draw_action = self.selected_idx_for_action(self.action_key, self.losses)

        return self.hp_fns[self.idx_draw_action]["best_action_fn"](params[self.idx_draw_action], state)

    @partial(jax.jit, static_argnames="self")
    def selected_idx_for_action(self, key, losses):
        key, choice_key = jax.random.split(key)
        p = jnp.nan_to_num(1 / (losses + 1e-6))

        selected_idx = jax.random.choice(choice_key, jnp.arange(self.n_networks), (), p=p / p.sum())

        return key, selected_idx

    def get_model(self) -> Dict:
        model = {}

        for idx_hp in range(self.n_networks):
            model[f"model_{idx_hp}"] = {
                "params": self.params[idx_hp],
                "hp_detail": self.hp_details[idx_hp],
            }

        model["idx_compute_target"] = jnp.nanargmin(self.losses)

        return model
