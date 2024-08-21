from typing import Tuple, List, Callable, Dict
from functools import partial
import jax
import jax.numpy as jnp
import optax
from dehb import DEHB
from ConfigSpace import ConfigurationSpace, Integer, Float, Configuration
from slimdqn.networks.base_dqn import BaseDQN


class HPGenerator:
    def __init__(self, key, observation_dim, n_actions, hp_space) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.hp_space = hp_space

        self.config_space = ConfigurationSpace(
            seed=int(key[1]),
            space={
                "n_layers": Integer("n_layers", bounds=(hp_space["n_layers_range"][0], hp_space["n_layers_range"][1])),
                "n_neurons": Integer(
                    "n_neurons", bounds=(hp_space["n_neurons_range"][0], hp_space["n_neurons_range"][1])
                ),
                "idx_activation": Integer("idx_activation", bounds=(0, len(hp_space["activations"]) - 1)),
                "idx_loss": Integer("idx_loss", bounds=(0, len(hp_space["losses"]) - 1)),
                "idx_optimizer": Integer("idx_optimizer", bounds=(0, len(hp_space["optimizers"]) - 1)),
                "learning_rate": Float(
                    "learning_rate",
                    bounds=(10 ** hp_space["learning_rate_range"][0], 10 ** hp_space["learning_rate_range"][1]),
                    log=True,
                ),
            },
        )

    def sample(self, key):
        hp_detail = dict(self.config_space.sample_configuration())

        q = BaseDQN(
            [hp_detail["n_neurons"]] * hp_detail["n_layers"],
            [self.hp_space["activations"][hp_detail["idx_activation"]]] * hp_detail["n_layers"],
            self.hp_space["cnn"],
            self.n_actions,
            self.hp_space["losses"][hp_detail["idx_loss"]],
        )
        params = q.q_network.init(key, jnp.zeros(self.observation_dim))
        optimizer = self.hp_space["optimizers"][hp_detail["idx_optimizer"]](hp_detail["learning_rate"])
        optimizer_state = optimizer.init(params)

        def update_and_loss_fn(params, targets, batch_samples, optimizer_state):
            loss, grad_loss = q.value_and_grad(params, targets, batch_samples)
            updates, optimizer_state = optimizer.update(grad_loss, optimizer_state, params)
            params = optax.apply_updates(params, updates)

            return loss, params, optimizer_state

        hp_fn = {
            "apply_fn": jax.jit(q.q_network.apply),
            "update_and_loss_fn": jax.jit(update_and_loss_fn),
            "best_action_fn": jax.jit(q.best_action),
        }

        return hp_fn, params, optimizer_state, hp_detail
