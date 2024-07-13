from typing import Dict
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
from slimRL.networks.singleDQN import SingleDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class DQN:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        hidden_layers: list,
        lr: float,
        gamma: float,
        update_horizon: int,
        train_frequency: int,
        target_update_frequency: int,
        loss_type: str,
    ):
        self.q = SingleDQN(key, observation_dim, n_actions, hidden_layers, loss_type)
        self.params = self.q.params
        self.target_params = self.q.params

        self.optimizer = optax.adam(lr)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

    def update_online_params(self, step: int, key, batch_size: int, replay_buffer: ReplayBuffer):
        if step % self.train_frequency == 0:
            batch_samples = replay_buffer.sample_transition_batch(batch_size, key)

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params,
                self.target_params,
                self.optimizer_state,
                batch_samples,
            )

            return loss
        return jnp.nan

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        target_params: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        value_next_states = self.q.apply(target_params, batch_samples["next_observations"])
        targets = self.compute_target_from_values(value_next_states, batch_samples)

        loss, grad_loss = self.q.value_and_grad(params, targets, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

    def compute_target_from_values(self, value_next_states, batch_samples):
        # computes the target value for single or a batch of samples
        return batch_samples["rewards"] + (1 - batch_samples["dones"]) * self.gamma * jnp.max(
            value_next_states, axis=-1
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):  # computes the best action for a single state
        return jnp.argmax(self.q.apply(params, state)).astype(jnp.int8)

    def get_model(self) -> Dict:
        return {
            "params": self.params,
            "hidden_layers": self.q.q_network.hidden_layers,
        }
