from typing import Sequence, Callable
import jax.numpy as jnp
import flax.linen as nn


class DQNNet(nn.Module):
    features: Sequence[int]
    activations: Sequence[Callable]
    n_actions: int
    cnn: bool

    def setup(self):
        self.range_idx_mlp_layers = range(3 if self.cnn else 0, len(self.features))

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")

        if self.cnn:
            x = nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                x / 255.0
            )
            x = self.activations[0](x)
            x = nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
            x = self.activations[1](x)
            x = nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
            x = self.activations[2](x).flatten()
        else:
            x = jnp.squeeze(x)

        for idx_layer in self.range_idx_mlp_layers:
            x = self.activations[idx_layer](nn.Dense(self.features[idx_layer], kernel_init=initializer)(x))
        x = nn.Dense(self.n_actions, kernel_init=initializer)(x)
        return x
