from typing import Callable, List

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection import IDX_RB


class BaseDQN:
    def __init__(
        self,
        cnn_n_layers: int,
        cnn_n_channels: int,
        cnn_kernel_size: int,
        cnn_stride: int,
        mlp_n_layers: int,
        mlp_n_neurons: int,
        activation: Callable,
        n_actions: int,
        loss: Callable,
    ):
        self.q_network = DQNNet(
            cnn_n_layers,
            cnn_n_channels,
            cnn_kernel_size,
            cnn_stride,
            mlp_n_layers,
            mlp_n_neurons,
            activation,
            n_actions,
        )
        self.loss = loss

    def value_and_grad(self, params: FrozenDict, targets, samples):
        (_, loss_l2), grad = jax.value_and_grad(self.loss_from_targets, has_aux=True)(params, targets, samples)

        return loss_l2, grad

    def loss_from_targets(self, params: FrozenDict, targets, samples):
        q_values = jax.vmap(lambda state, action: self.q_network.apply(params, state)[action])(
            samples[IDX_RB["state"]], samples[IDX_RB["action"]]
        )

        # always return the l2 loss to compare fairly between the networks
        return self.loss(q_values, targets).mean(), optax.l2_loss(q_values, targets).mean()

    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.q_network.apply(params, state)).astype(jnp.int8)
