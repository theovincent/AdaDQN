from typing import Dict
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class SingleDQN:
    def update_online_params(self, step: int, key: jax.Array, batch_size: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch(batch_size, key)

            loss = self.learn_on_batch(batch_samples)

            return loss
        return jnp.nan

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

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

        return {
            "params": self.params,
            "optimizer_hps": self.hyperparameters_fn["optimizer_hps"],
            "architecture_hps": self.hyperparameters_fn["architecture_hps"],
        }
