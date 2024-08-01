from typing import Callable, List
import optax
import jax
from flax.core import FrozenDict
from functools import partial
import jax.numpy as jnp
from slimRL.networks.architectures.dqn_net import DQNNet

from slimRL.networks import IDX_RB


class BaseDQN:
    def __init__(self, n_actions: int, features: list, activations: List, cnn: bool, loss: Callable):
        self.q_network = DQNNet(features, activations, n_actions, cnn)
        self.loss = loss

    @partial(jax.jit, static_argnames="self")
    def value_and_grad(self, params: FrozenDict, targets, samples):
        (_, loss_l2), grad = jax.value_and_grad(self.loss_from_targets, has_aux=True)(params, targets, samples)

        return loss_l2, grad

    def loss_from_targets(self, params: FrozenDict, targets, samples):
        q_values = jax.vmap(lambda state, action: self.apply(params, state)[action])(
            samples[IDX_RB["state"]], samples[IDX_RB["action"]]
        )

        # alwars return the l2 loss to compare fairly between the networks
        return self.loss(q_values, targets).mean(), optax.l2_loss(q_values, targets).mean()

    @partial(jax.jit, static_argnames="self")
    def apply(self, params: FrozenDict, states: jnp.ndarray):
        # computes the q values for single or batch of states
        return self.q_network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.apply(params, state)).astype(jnp.int8)
