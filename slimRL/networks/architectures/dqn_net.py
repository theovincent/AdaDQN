from typing import Sequence, Callable
import jax.numpy as jnp
import flax.linen as nn


class DQNNet(nn.Module):
    hidden_layers: Sequence[int]
    activations: Sequence[Callable]
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = jnp.squeeze(x)
        for idx_layer in range(len(self.hidden_layers)):
            x = self.activations[idx_layer](nn.Dense(self.hidden_layers[idx_layer])(x))
        x = nn.Dense(self.n_actions)(x)
        return x
