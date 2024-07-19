from typing import Dict, Tuple, List, Callable
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
from slimRL.networks.hyperparameter_generator import DEHBGenerator
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class DEHBDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        optimizers: List[Callable],
        lr_range: Tuple[int],
        losses: List[str],
        n_layers_range: Tuple[int],
        n_neurons_range: Tuple[int],
        activations: List[Callable],
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        min_n_epochs_per_hypeparameter: int,
        max_n_epochs_per_hypeparameter: int,
    ):
        self.q_key, dehb_key = jax.random.split(key)
        self.hyperparameters_generator = DEHBGenerator(
            dehb_key,
            observation_dim,
            n_actions,
            optimizers,
            lr_range,
            losses,
            n_layers_range,
            n_neurons_range,
            activations,
            min_n_epochs_per_hypeparameter,
            max_n_epochs_per_hypeparameter,
        )

        self.q_key, hp_key = jax.random.split(key)
        self.hyperparameters_fn, self.params, self.optimizer_state, self.n_epochs_per_hypeparameter = (
            self.hyperparameters_generator(hp_key, None, None, None)
        )

        self.hyperparameters_details = {
            "optimizer_hps": [self.hyperparameters_fn["optimizer_hps"]],
            "architecture_hps": [self.hyperparameters_fn["architecture_hps"]],
        }
        print(f"Starting optimizer: {self.hyperparameters_fn['optimizer_hps']}", flush=True)
        print(f"and architecture: {self.hyperparameters_fn['architecture_hps']}", flush=True)
        print(f"and n_epochs_per_hypeparameter: {self.n_epochs_per_hypeparameter}", end="\n\n", flush=True)

        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency

    def update_online_params(self, step: int, key: jax.Array, batch_size: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch(batch_size, key)

            loss = self.learn_on_batch(batch_samples)

            return loss
        return jnp.nan

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

    def update_hyperparamters(self, idx_epoch, avg_return):
        if idx_epoch % self.n_epochs_per_hypeparameter == 0:
            self.q_key, hp_key = jax.random.split(self.q_key)
            self.hyperparameters_fn, self.params, self.optimizer_state, self.n_epochs_per_hypeparameter = (
                self.hyperparameters_generator(
                    hp_key, self.hyperparameters_fn, avg_return, self.n_epochs_per_hypeparameter
                )
            )

            self.hyperparameters_details["optimizer_hps"].append(self.hyperparameters_fn["optimizer_hps"])
            print(
                f"\nChange optimizer: {self.hyperparameters_details['optimizer_hps'][-2]} for {self.hyperparameters_fn['optimizer_hps']}",
                flush=True,
            )
            self.hyperparameters_details["architecture_hps"].append(self.hyperparameters_fn["architecture_hps"])
            print(
                f"and change architecture: {self.hyperparameters_details['architecture_hps'][-2]} for {self.hyperparameters_fn['architecture_hps']}",
                flush=False,
            )
            print(f"and n_epochs_per_hypeparameter: {self.n_epochs_per_hypeparameter}", flush=False)

    def learn_on_batch(self, batch_samples):
        value_next_states = self.hyperparameters_fn["apply_fn"](self.target_params, batch_samples["next_observations"])
        targets = self.compute_target_from_values(value_next_states, batch_samples)

        loss, grad_loss = self.hyperparameters_fn["grad_and_loss_fn"](self.params, targets, batch_samples)
        updates, self.optimizer_state = self.hyperparameters_fn["optimizer_fn"](
            grad_loss, self.optimizer_state, self.params
        )
        self.params = jax.jit(optax.apply_updates)(self.params, updates)

        return loss

    @partial(jax.jit, static_argnames="self")
    def compute_target_from_values(self, value_next_states, batch_samples):
        # computes the target value for single or a batch of samples
        return batch_samples["rewards"] + (1 - batch_samples["dones"]) * self.gamma**self.update_horizon * jnp.max(
            value_next_states, axis=-1
        )

    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return self.hyperparameters_fn["best_action_fn"](params, state)

    def get_model(self) -> Dict:
        model = {
            "params": self.params,
            "optimizer_hps": self.hyperparameters_fn["optimizer_hps"],
            "architecture_hps": self.hyperparameters_fn["architecture_hps"],
        }

        return model
