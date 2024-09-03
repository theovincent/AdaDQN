import unittest
import jax
import numpy as np
import copy

from slimdqn.networks.hyperparameters.generators import HPGenerator

from slimdqn.networks import ACTIVATIONS, LOSSES, OPTIMIZERS


class TestLunarLander(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hp_space = {
            "cnn_n_layers_range": [2, 10],
            "cnn_n_channels_range": [16, 64],
            "cnn_kernel_size_range": [2, 8],
            "cnn_stride_range": [1, 8],
            "mlp_n_layers_range": [1, 3],
            "mlp_n_neurons_range": [50, 200],
            "activations": list(ACTIVATIONS.values()),
            "losses": list(LOSSES.values()),
            "optimizers": list(OPTIMIZERS.values()),
            "learning_rate_range": [6, 2],
        }
        self.hp_generator = HPGenerator(jax.random.PRNGKey(0), (10, 10, 4), 7, hp_space, "elitsm")

        _, self.params, _, self.hp_detail = self.hp_generator.sample(jax.random.PRNGKey(0))

    def test_add_cnn_layer(self):
        print(self.hp_detail["cnn_n_layers"] + 1)
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_cnn_n_layers = np.clip(
            self.hp_detail["cnn_n_layers"] + 1, *self.hp_generator.hp_space["cnn_n_layers_range"]
        )

        assert new_hp_detail["cnn_n_layers"] == new_cnn_n_layers, "No CNN layer has been added."
        assert f"Conv_{new_cnn_n_layers - 1}" in new_params["params"].keys(), "No CNN layer has been added."

        try:
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(0), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_cnn_layer(self):
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_cnn_n_layers = np.clip(
            self.hp_detail["cnn_n_layers"] - 1, *self.hp_generator.hp_space["cnn_n_layers_range"]
        )

        assert new_hp_detail["cnn_n_layers"] == new_cnn_n_layers, "No CNN layer has been removed."
        assert f"Conv_{new_cnn_n_layers}" not in new_params["params"].keys(), "No CNN layer has been removed."

        try:
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(0), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"
