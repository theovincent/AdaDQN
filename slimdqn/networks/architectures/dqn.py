from typing import Callable

import flax.linen as nn
import jax.numpy as jnp


class DQNNet(nn.Module):
    cnn_n_layers: int
    cnn_n_channels: int
    cnn_kernel_size: int
    cnn_stride: int
    mlp_n_layers: int
    mlp_n_neurons: int
    activation: Callable
    n_actions: int

    @nn.compact
    def __call__(self, x):
        if self.cnn_n_layers == 0:
            initializer = nn.initializers.lecun_normal()
        else:
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
            x = jnp.array(x, ndmin=4) / 255.0

        for _ in range(self.cnn_n_layers):
            x = self.activation(
                nn.Conv(
                    features=self.cnn_n_channels,
                    kernel_size=(self.cnn_kernel_size, self.cnn_kernel_size),
                    strides=(self.cnn_stride, self.cnn_stride),
                    kernel_init=initializer,
                )(x)
            )
        if self.cnn_n_layers > 0:
            x = x.reshape((x.shape[0], -1))

        x = jnp.squeeze(x)

        for _ in range(self.mlp_n_layers):
            x = self.activation((nn.Dense(self.mlp_n_neurons, kernel_init=initializer)(x)))

        return nn.Dense(self.n_actions, kernel_init=initializer)(x)
