from typing import Dict, List, Callable, Tuple
import jax
import jax.numpy as jnp
from slimRL.networks.single_dqn import SingleDQN
from slimRL.networks.base_dqn import BaseDQN


class DQN(SingleDQN):
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        optimizer: Callable,
        learning_rate: float,
        loss: Callable,
        features: Tuple[int],
        activations: List[Callable],
        cnn: bool,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
    ):
        print(
            f"Start training with: {optimizer.func.__name__}, lr = {learning_rate}, {loss.__name__}, features = {features} and {[activation.__name__ for activation in activations]}.",
            flush=True,
        )

        self.hyperparameters_fn = {}

        optimizer = optimizer(learning_rate)
        self.hyperparameters_fn["optimizer_fn"] = jax.jit(optimizer.update)

        q = BaseDQN(n_actions, features, activations, cnn, loss)

        self.hyperparameters_fn["apply_fn"] = q.apply
        self.hyperparameters_fn["grad_and_loss_fn"] = q.value_and_grad
        self.hyperparameters_fn["best_action_fn"] = q.best_action
        self.params = q.q_network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer_state = optimizer.init(self.params)

        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency

    def get_model(self) -> Dict:
        return {"params": self.params}
