from typing import List
import optax
import jax
from slimRL.networks.singleDQN import SingleDQN


class HyperparametersGenerator:
    def __init__(
        self, observation_dim, n_actions, n_layers_range: List, n_neurons_range: List, lr: float, loss_type: str
    ) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.n_layers_range = n_layers_range
        self.n_neurons_range = n_neurons_range
        self.lr = lr
        self.loss_type = loss_type

        self.fixed_hidden_layers = iter([[25, 25], [50, 50], [100, 100], [200, 200]])

    def __call__(self, key: jax.Array):
        key_layers, key_neurons, network_key = jax.random.split(key, 3)

        n_layers = jax.random.randint(key_layers, (), self.n_layers_range[0], self.n_layers_range[1] + 1)
        hidden_layers = list(
            jax.random.randint(key_neurons, (n_layers,), self.n_neurons_range[0], self.n_neurons_range[1] + 1)
        )

        print(f"New Q-Network with hidden layers: {hidden_layers}")
        dqn_network = SingleDQN(network_key, self.observation_dim, self.n_actions, hidden_layers, self.loss_type)

        optimizer = optax.adam(self.lr)
        optimizer_state = optimizer.init(dqn_network.params)

        hyperparameters_fn = {
            "apply_fn": dqn_network.apply,
            "grad_and_loss_fn": dqn_network.value_and_grad,
            "optimizer_fn": jax.jit(optimizer.update),
            "best_action_fn": dqn_network.best_action,
            "details": dqn_network.q_network.hidden_layers,
        }

        return hyperparameters_fn, dqn_network.params, optimizer_state

    def fixed_hypeparameter(self):
        dqn_network = SingleDQN(
            jax.random.PRNGKey(0), self.observation_dim, self.n_actions, next(self.fixed_hidden_layers), self.loss_type
        )

        optimizer = optax.adam(self.lr)
        optimizer_state = optimizer.init(dqn_network.params)

        hyperparameters_fn = {
            "apply_fn": dqn_network.apply,
            "grad_and_loss_fn": dqn_network.value_and_grad,
            "optimizer_fn": jax.jit(optimizer.update),
            "best_action_fn": dqn_network.best_action,
            "details": dqn_network.q_network.hidden_layers,
        }

        return hyperparameters_fn, dqn_network.params, optimizer_state
