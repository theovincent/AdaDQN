from functools import partial
import optax
import jax.numpy as jnp
import flax.linen as nn


ACTIVATIONS = {
    "celu": nn.celu,
    "elu": nn.elu,
    "gelu": nn.gelu,
    "hard_sigmoid": nn.hard_sigmoid,
    "hard_silu": nn.hard_silu,
    "hard_tanh": nn.hard_tanh,
    "leaky_relu": nn.leaky_relu,
    "log_sigmoid": nn.log_sigmoid,
    "log_softmax": nn.log_softmax,
    "relu": nn.relu,
    "selu": nn.selu,
    "sigmoid": nn.sigmoid,
    "silu": nn.silu,
    "soft_sign": nn.soft_sign,
    "softmax": nn.softmax,
    "softplus": nn.softplus,
    "standardize": nn.standardize,
    "tanh": nn.tanh,
}

OPTIMIZERS = {
    "adabelief": optax.adabelief,
    "adadelta": optax.adadelta,
    "adagrad": optax.adagrad,
    "adafactor": optax.adafactor,
    "adam": partial(optax.adam, eps=1.5e-4),
    "adam_medium_eps": partial(optax.adam, eps=1.5e-6),
    "adam_small_eps": partial(optax.adam, eps=1.5e-8),
    "adamax": optax.adamax,
    "adamaxw": optax.adamaxw,
    "adamw": optax.adamw,
    "amsgrad": optax.amsgrad,
    "fromage": optax.fromage,
    "lamb": optax.lamb,
    "lars": optax.lars,
    "lion": optax.lion,
    "nadam": optax.nadam,
    "nadamw": optax.nadamw,
    "noisy_sgd": optax.noisy_sgd,
    "novograd": optax.novograd,
    "optimistic_gradient_descent": optax.optimistic_gradient_descent,
    "radam": optax.radam,
    "rmsprop": optax.rmsprop,
    "rprop": optax.rprop,
    "sgd": optax.sgd,
    "sm3": optax.sm3,
    "yogi": optax.yogi,
}

LOSSES = {
    "huber": optax.huber_loss,
    "l1": lambda y_hat, y: jnp.abs(y_hat - y),
    "l2": optax.l2_loss,
    "log_cosh": optax.log_cosh,
}

IDX_RB = {
    "state": 0,
    "action": 1,
    "reward": 2,
    "next_state": 3,
    "next_action": 4,
    "next_reward": 5,
    "terminal": 6,
    "indices": 7,
}
