from typing import Dict
from typing import Tuple
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
import numpy as np
from slimRL.networks.hyperparameter_generator import HyperparametersGenerator
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class AdaDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_networks,
        n_layers_range: Tuple[int],
        n_neurons_range: Tuple[int],
        lr: float,
        gamma: float,
        update_horizon: int,
        train_frequency: int,
        target_update_frequency: int,
        loss_type: str,
        end_online_exp: float,
        duration_online_exp: int,
    ):
        self.q_key, self.action_key = jax.random.split(key)
        self.n_networks = n_networks
        self.hyperparameters_generator = HyperparametersGenerator(
            observation_dim, n_actions, n_layers_range, n_neurons_range, lr, loss_type
        )

        self.hyperparameters_fn = []
        self.params = []
        self.optimizer_state = []

        for _ in range(self.n_networks):
            self.q_key, hp_key = jax.random.split(self.q_key)
            # hyperparameters_fn, params, optimizer_state = self.hyperparameters_generator(hp_key)
            hyperparameters_fn, params, optimizer_state = self.hyperparameters_generator.fixed_hypeparameter()

            self.hyperparameters_fn.append(hyperparameters_fn)
            self.params.append(params)
            self.optimizer_state.append(optimizer_state)

        self.target_params = self.params[0]
        self.idx_compute_target = 0
        self.indexes_compute_target = []
        self.losses = np.zeros(n_networks)

        self.epsilon_b_schedule = optax.linear_schedule(1.0, end_online_exp, duration_online_exp)
        self.n_draws = 0
        self.idx_draw_action = 0
        self.indexes_draw_action = []

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

    def update_online_params(self, step: int, key: jax.Array, batch_size: int, replay_buffer: ReplayBuffer):
        if step % self.train_frequency == 0:
            batch_samples = replay_buffer.sample_transition_batch(batch_size, key)

            losses = self.learn_on_batch(batch_samples)
            self.losses += losses

            return np.mean(losses)
        return jnp.nan

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            # Define new target
            self.idx_compute_target = jnp.argmin(self.losses)
            self.target_params = self.params[self.idx_compute_target].copy()

            # Change worst online network
            # idx_new_hyperparameter = jnp.argmax(self.losses)

            # self.q_key, hp_key = jax.random.split(self.q_key)
            # hyperparameters_fn, params, optimizer_state = self.hyperparameters_generator(hp_key)

            # self.hyperparameters_fn[idx_new_hyperparameter] = hyperparameters_fn
            # self.params[idx_new_hyperparameter] = params
            # self.optimizer_state[idx_new_hyperparameter] = optimizer_state

            # Reset the loss
            self.indexes_compute_target.append(self.idx_compute_target)
            self.indexes_draw_action.append(self.idx_draw_action)
            self.losses = np.zeros_like(self.losses)

    def learn_on_batch(self, batch_samples):
        losses = np.zeros(self.n_networks)

        value_next_states = self.hyperparameters_fn[self.idx_compute_target]["apply_fn"](
            self.target_params, batch_samples["next_observations"]
        )
        targets = self.compute_target_from_values(value_next_states, batch_samples)

        for idx_hyperparameter in range(self.n_networks):
            loss, grad_loss = self.hyperparameters_fn[idx_hyperparameter]["grad_and_loss_fn"](
                self.params[idx_hyperparameter], targets, batch_samples
            )
            updates, self.optimizer_state[idx_hyperparameter] = self.hyperparameters_fn[idx_hyperparameter][
                "optimizer_fn"
            ](grad_loss, self.optimizer_state[idx_hyperparameter])
            self.params[idx_hyperparameter] = jax.jit(optax.apply_updates)(self.params[idx_hyperparameter], updates)

            losses[idx_hyperparameter] = loss

        return losses

    @partial(jax.jit, static_argnames="self")
    def compute_target_from_values(self, value_next_states, batch_samples):
        # computes the target value for single or a batch of samples
        return batch_samples["rewards"] + (1 - batch_samples["dones"]) * self.gamma * jnp.max(
            value_next_states, axis=-1
        )

    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        self.action_key, self.idx_draw_action = self.selected_idx_for_action(
            self.action_key, self.n_draws, self.idx_compute_target
        )
        self.n_draws += 1

        return self.hyperparameters_fn[self.idx_draw_action]["best_action_fn"](params[self.idx_draw_action], state)

    @partial(jax.jit, static_argnames="self")
    def selected_idx_for_action(self, key, n_draws, idx_compute_target):
        key, epsilon_key, sample_key = jax.random.split(key, 3)

        selected_idx = jax.lax.select(
            jax.random.uniform(epsilon_key) < self.epsilon_b_schedule(n_draws),
            jax.random.randint(sample_key, (), 0, self.n_networks),  # if true
            idx_compute_target,  # if false
        )

        return key, selected_idx

    def get_model(self) -> Dict:
        model = {}

        for idx_hp, hyperparameter in enumerate(self.hyperparameters_fn):
            model[f"model_{idx_hp}"] = {
                "params": self.params[idx_hp],
                "details": hyperparameter["details"],
            }

        model["idx_compute_target"] = jnp.argmin(self.losses)

        return model
