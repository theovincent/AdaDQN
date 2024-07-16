from typing import Tuple, List, Callable, Dict
from functools import partial
import optax
import jax
import jax.numpy as jnp
from slimRL.networks.singleDQN import SingleDQN


class HyperparametersGenerator:
    def __init__(
        self,
        observation_dim,
        n_actions,
        n_layers_range: Tuple,
        n_neurons_range: Tuple,
        activations: List[Callable],
        lr_range: Tuple,
        optimizers: List[Callable],
        losses: List[Callable],
    ) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions

        self.n_layers_range = n_layers_range
        self.n_neurons_range = n_neurons_range
        self.activations = activations
        self.lr_range = lr_range
        self.optimizers = optimizers
        self.losses = losses

        self.dummy_hyperparameters_fn = {
            "optimizer_hps": {"learning_rate": 0.0, "idx_optimizer": 0},
            "architecture_hps": {},
        }

    def __call__(
        self, key: jax.Array, hyperparameters_fn: Callable, params: Dict, optimizer_state: Dict, force_new: bool
    ):
        key, optimizer_key, architecture_key = jax.random.split(key, 3)

        change_optimizer, hyperparameters_fn["optimizer_hps"] = self.change_optimizer_hps(
            optimizer_key, hyperparameters_fn["optimizer_hps"], force_new
        )
        if change_optimizer:
            optimizer = self.optimizers[hyperparameters_fn["optimizer_hps"]["idx_optimizer"]](
                hyperparameters_fn["optimizer_hps"]["learning_rate"]
            )
            hyperparameters_fn["optimizer_fn"] = jax.jit(optimizer.update)

            # We change the architecture only if we change the optimizer
            change_architecture, n_layers, idx_loss = self.change_architecture_hps(architecture_key, force_new)

            if change_architecture:
                neurons_key, activation_key, init_key = jax.random.split(key, 3)
                hyperparameters_fn["architecture_hps"]["hidden_layers"] = list(
                    jax.random.randint(neurons_key, (n_layers,), self.n_neurons_range[0], self.n_neurons_range[1] + 1)
                )
                hyperparameters_fn["architecture_hps"]["indices_activations"] = list(
                    jax.random.randint(activation_key, (n_layers,), 0, len(self.activations))
                )
                hyperparameters_fn["architecture_hps"]["idx_loss"] = idx_loss

                q = SingleDQN(
                    self.n_actions,
                    hyperparameters_fn["architecture_hps"]["hidden_layers"],
                    [self.activations[idx] for idx in hyperparameters_fn["architecture_hps"]["indices_activations"]],
                    self.losses[hyperparameters_fn["architecture_hps"]["idx_loss"]],
                )

                hyperparameters_fn["apply_fn"] = q.apply
                hyperparameters_fn["grad_and_loss_fn"] = q.value_and_grad
                hyperparameters_fn["best_action_fn"] = q.best_action
                params = q.q_network.init(init_key, jnp.zeros(self.observation_dim, dtype=jnp.float32))

            optimizer_state = optimizer.init(params)
        else:
            change_architecture = False

        return hyperparameters_fn, params, optimizer_state, change_optimizer, change_architecture

    @partial(jax.jit, static_argnames="self")
    def change_optimizer_hps(self, key, optimizer_hps, force_new):
        change_key, generate_hp_key = jax.random.split(key)

        change_optimizer = jnp.logical_or(jax.random.bernoulli(change_key, p=0.02), force_new)
        optimizer_hps = jax.lax.cond(
            change_optimizer,
            self.generate_hp_optimizer,
            lambda key, hp_: hp_,
            generate_hp_key,
            optimizer_hps,
        )

        return change_optimizer, optimizer_hps

    def generate_hp_optimizer(self, key, optimizer_hps):
        lr_key, idx_key = jax.random.split(key)

        # sample the learning rate in log space
        optimizer_hps["learning_rate"] = 10 ** jax.random.uniform(
            lr_key, minval=self.lr_range[0], maxval=self.lr_range[1]
        )
        optimizer_hps["idx_optimizer"] = jax.random.randint(idx_key, (), minval=0, maxval=len(self.optimizers))

        return optimizer_hps

    @partial(jax.jit, static_argnames="self")
    def change_architecture_hps(self, key, force_new):
        change_key, generate_hp_key = jax.random.split(key)

        change_architecture = jnp.logical_or(jax.random.bernoulli(change_key, p=0.7), force_new)
        n_layers, idx_loss = jax.lax.cond(
            change_architecture, self.generate_hp_architecture, lambda key: (0, 0), generate_hp_key
        )

        return change_architecture, n_layers, idx_loss

    def generate_hp_architecture(self, key):
        layers_key, loss_key = jax.random.split(key)

        n_layers = jax.random.randint(layers_key, (), self.n_layers_range[0], self.n_layers_range[1] + 1)
        idx_loss = jax.random.randint(loss_key, (), minval=0, maxval=len(self.losses))

        return n_layers, idx_loss
