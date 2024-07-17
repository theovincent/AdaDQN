from typing import Dict, Tuple, List, Callable
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
import numpy as np
from slimRL.networks.hyperparameter_generator import HyperparametersGenerator
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class RSDQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_layers_range: Tuple[int],
        n_neurons_range: Tuple[int],
        activations: List[Callable],
        lr_range: Tuple[int],
        optimizers: List[Callable],
        losses: str,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        # To make sure AdaDQN samples the same hyperparameters
        self.q_key, _ = jax.random.split(key)
        self.hyperparameters_generator = HyperparametersGenerator(
            observation_dim, n_actions, n_layers_range, n_neurons_range, activations, lr_range, optimizers, losses
        )

        self.q_key, hp_key = jax.random.split(self.q_key)
        self.hyperparameters_fn, self.params, self.optimizer_state, _, _ = self.hyperparameters_generator(
            hp_key, self.hyperparameters_generator.dummy_hyperparameters_fn, None, None, force_new=True
        )

        self.hyperparameters_details = {"optimizer_hps": [], "architecture_hps": []}
        slim_optimizer_hps = jax.tree_map(lambda obj: obj.item(), self.hyperparameters_fn["optimizer_hps"])
        self.hyperparameters_details["optimizer_hps"].append([slim_optimizer_hps])
        print(f"Starting optimizer: {slim_optimizer_hps}", flush=True)
        slim_architecture_hps = jax.tree_map(lambda obj: obj.item(), self.hyperparameters_fn["architecture_hps"])
        self.hyperparameters_details["architecture_hps"].append([slim_architecture_hps])
        print(f"and architecture: {slim_architecture_hps}", end="\n\n", flush=True)

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

            self.q_key, hp_key = jax.random.split(self.q_key)
            (
                self.hyperparameters_fn,
                self.params,
                self.optimizer_state,
                change_optimizer,
                change_architecture,
            ) = self.hyperparameters_generator(
                hp_key,
                self.hyperparameters_fn,
                self.params,
                self.optimizer_state,
                force_new=False,
            )

            if change_optimizer:
                slim_optimizer_hps = jax.tree_map(lambda obj: obj.item(), self.hyperparameters_fn["optimizer_hps"])
                self.hyperparameters_details["optimizer_hps"].append(slim_optimizer_hps)
                print(
                    f"\nChange optimizer: {self.hyperparameters_details['optimizer_hps'][-2]} for {slim_optimizer_hps}",
                    flush=True,
                )

                if change_architecture:
                    slim_architecture_hps = jax.tree_map(
                        lambda obj: obj.item(), self.hyperparameters_fn["architecture_hps"]
                    )
                    self.hyperparameters_details["architecture_hps"].append(slim_architecture_hps)
                    print(
                        f"and change architecture: {self.hyperparameters_details['architecture_hps'][-2]} for {slim_architecture_hps}",
                        flush=True,
                    )

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
