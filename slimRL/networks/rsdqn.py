from typing import List, Callable, Tuple
import jax
from slimRL.networks.single_dqn import SingleDQN
from slimRL.networks.hyperparameter_generator import RandomGenerator


class RSDQN(SingleDQN):
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        optimizers: List[Callable],
        learning_rate_range: Tuple[int],
        losses: List[Callable],
        n_layers_range: Tuple[int],
        n_neurons_range: Tuple[int],
        activations: List[Callable],
        cnn: bool,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        n_epochs_per_hyperparameter: int,
    ):
        self.hyperparameters_generator = RandomGenerator(
            observation_dim,
            n_actions,
            optimizers,
            learning_rate_range,
            losses,
            n_layers_range,
            n_neurons_range,
            activations,
            cnn,
            1.0,
            1.0,
        )

        self.q_key, hp_key = jax.random.split(key)
        self.hyperparameters_fn, self.params, self.optimizer_state, _, _ = self.hyperparameters_generator(
            hp_key, self.hyperparameters_generator.dummy_hyperparameters_fn, None, None, force_new=True
        )

        self.hyperparameters_details = {
            "optimizer_hps": [self.hyperparameters_fn["optimizer_hps"]],
            "architecture_hps": [self.hyperparameters_fn["architecture_hps"].copy()],
        }
        print(f"Starting optimizer: {self.hyperparameters_fn['optimizer_hps']}", flush=True)
        print(f"and architecture: {self.hyperparameters_fn['architecture_hps']}", end="\n\n", flush=True)

        self.target_params = self.params.copy()

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.n_epochs_per_hyperparameter = n_epochs_per_hyperparameter

    def update_hyperparamters(self, idx_epoch, *args):
        if idx_epoch % self.n_epochs_per_hyperparameter == 0:
            self.q_key, hp_key = jax.random.split(self.q_key)
            self.hyperparameters_fn, self.params, self.optimizer_state, _, _ = self.hyperparameters_generator(
                hp_key, self.hyperparameters_generator.dummy_hyperparameters_fn, None, None, force_new=True
            )

            self.target_params = self.params.copy()

            self.hyperparameters_details["optimizer_hps"].append(self.hyperparameters_fn["optimizer_hps"])
            print(
                f"\nChange optimizer: {self.hyperparameters_details['optimizer_hps'][-2]} for {self.hyperparameters_fn['optimizer_hps']}",
                flush=True,
            )
            self.hyperparameters_details["architecture_hps"].append(self.hyperparameters_fn["architecture_hps"].copy())
            print(
                f"and change architecture: {self.hyperparameters_details['architecture_hps'][-2]} for {self.hyperparameters_fn['architecture_hps']}",
                flush=False,
            )
