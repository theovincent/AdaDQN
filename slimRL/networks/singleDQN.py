import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
from slimRL.networks.architectures.DQN import DQNNet


class SingleDQN:
    def __init__(self, init_key: jax.random.PRNGKey, observation_dim, n_actions, hidden_layers: list, loss_type: str):
        self.q_network = DQNNet(hidden_layers, n_actions)
        self.params = self.q_network.init(init_key, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.loss_type = loss_type

    @partial(jax.jit, static_argnames="self")
    def value_and_grad(self, params: FrozenDict, targets, samples):
        return jax.value_and_grad(self.loss_from_targets)(params, targets, samples)

    def loss_from_targets(self, params: FrozenDict, targets, samples):
        return jax.vmap(self.loss, in_axes=(None, 0, 0))(params, targets, samples).mean()

    def loss(self, params: FrozenDict, target, sample):
        # computes the loss for a single sample
        q_value = self.apply(params, sample["observations"])[sample["actions"]]
        return self.metric(q_value - target, ord=self.loss_type)

    @staticmethod
    def metric(error: jnp.ndarray, ord: str):
        if ord == "huber":
            return optax.huber_loss(error, 0)
        elif ord == "1":
            return jnp.abs(error)
        elif ord == "2":
            return jnp.square(error)

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray):
        # computes the q values for single or batch of states
        return self.q_network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.apply(params, state)).astype(jnp.int8)
