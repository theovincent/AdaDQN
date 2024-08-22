from collections import Counter
from typing import Tuple, List, Callable, Dict
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dehb import DEHB
from ConfigSpace import ConfigurationSpace, Integer, Float, Configuration
from slimdqn.networks.base_dqn import BaseDQN


class HPGenerator:
    def __init__(self, key, observation_dim, n_actions, hp_space, exploitation_type) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.hp_space = hp_space
        self.exploitation_type = exploitation_type

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

    def exploit_and_explore(self, key, metrics, hp_fns, params, optimizer_states, hp_details):
        n_networks = len(metrics)
        # exploit
        if self.exploitation_type == "elitism":
            selected_indices = [np.argmax(metrics)]
            for _ in range(n_networks - 1):
                key, selection_key = jax.random.split(key)
                selected_indices = jax.random.uniform(selection_key, (3,), int, 0, n_networks)
                selected_indices.append(selected_indices[np.argmax(metrics[selected_indices])])

            selected_indices_counter = Counter(selected_indices)

            indices_replacing_hps = []
            indices_new_hps = []
            for idx in range(n_networks):
                # if the idx has not been selected it will be replaced
                if idx not in selected_indices:
                    indices_new_hps.append(idx)
                # if the idx has been selected more that once (- 1), it should be added to the list of replacing idx
                else:
                    indices_replacing_hps.extend([idx] * (selected_indices_counter[idx] - 1))

        # explore
        for idx in range(len(indices_new_hps)):
            # change indices_replacing_hps[idx] hyperparameters and set it to indices_new_hps[idx]
            pass

        return indices_new_hps, hp_fns, params, optimizer_states, hp_details
