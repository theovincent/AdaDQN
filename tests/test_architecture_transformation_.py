import copy
import unittest

import jax
import numpy as np

from slimdqn.networks import ACTIVATIONS, LOSSES, OPTIMIZERS
from slimdqn.networks.hyperparameters.generators import HPGenerator


class TestTransformations(unittest.TestCase):
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
            "reset_weights": False,
        }
        self.np_key = np.random.randint(1000)
        self.hp_generator = HPGenerator(jax.random.PRNGKey(self.np_key), (10, 10, 4), 7, hp_space, "elitsm")

        _, self.params, _, self.hp_detail = self.hp_generator.sample(jax.random.PRNGKey(self.np_key))

    def test_add_cnn_layer(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_cnn_n_layers = np.clip(
            self.hp_detail["cnn_n_layers"] + 1, *self.hp_generator.hp_space["cnn_n_layers_range"]
        )

        assert new_hp_detail["cnn_n_layers"] == new_cnn_n_layers, "No CNN layer has been added."
        if "layers_to_skip" in new_params.keys():
            assert f"Conv_{new_cnn_n_layers - 1}" in new_params["layers_to_skip"], "No CNN layer has been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_cnn_layer(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_cnn_n_layers = np.clip(
            self.hp_detail["cnn_n_layers"] - 1, *self.hp_generator.hp_space["cnn_n_layers_range"]
        )

        assert new_hp_detail["cnn_n_layers"] == new_cnn_n_layers, "No CNN layer has been removed."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_layer(new_hp_detail, new_params, -1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_add_cnn_channels(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_channels(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_cnn_n_channels = np.clip(
            self.hp_detail["cnn_n_channels"] + 4 * 1, *self.hp_generator.hp_space["cnn_n_channels_range"]
        )

        assert new_hp_detail["cnn_n_channels"] == new_cnn_n_channels, "No CNN channels have been added."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN channels have been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_channels(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_cnn_channels(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_channels(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_cnn_n_channels = np.clip(
            self.hp_detail["cnn_n_channels"] - 4 * 1, *self.hp_generator.hp_space["cnn_n_channels_range"]
        )

        assert new_hp_detail["cnn_n_channels"] == new_cnn_n_channels, "No CNN channels have been added."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN channels have been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_channels(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_add_cnn_kernel_size(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_kernel_size(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_cnn_kernel_size = np.clip(
            self.hp_detail["cnn_kernel_size"] + 1, *self.hp_generator.hp_space["cnn_kernel_size_range"]
        )

        assert new_hp_detail["cnn_kernel_size"] == new_cnn_kernel_size, "No CNN kernel size has been increased."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN kernel size has been increased."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_kernel_size(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_cnn_kernel_size(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_kernel_size(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_cnn_kernel_size = np.clip(
            self.hp_detail["cnn_kernel_size"] - 1, *self.hp_generator.hp_space["cnn_kernel_size_range"]
        )

        assert new_hp_detail["cnn_kernel_size"] == new_cnn_kernel_size, "No CNN kernel size has been reduced."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN kernel size has been reduced."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_kernel_size(new_hp_detail, new_params, -1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_add_cnn_stride(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_stride(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_cnn_stride = np.clip(self.hp_detail["cnn_stride"] + 1, *self.hp_generator.hp_space["cnn_stride_range"])

        assert new_hp_detail["cnn_stride"] == new_cnn_stride, "No CNN stride has been increased."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN stride have been increased."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_stride(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_cnn_stride(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_cnn_stride(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_cnn_stride = np.clip(self.hp_detail["cnn_stride"] - 1, *self.hp_generator.hp_space["cnn_stride_range"])

        assert new_hp_detail["cnn_stride"] == new_cnn_stride, "No CNN stride has been increased."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [layer_key.startswith("Dense") for layer_key in new_params["layers_to_skip"]]
            ), "No CNN stride have been increased."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_cnn_stride(new_hp_detail, new_params, -1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_add_mlp_layer(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_mlp_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_mlp_n_layers = np.clip(
            self.hp_detail["mlp_n_layers"] + 1, *self.hp_generator.hp_space["mlp_n_layers_range"]
        )

        assert new_hp_detail["mlp_n_layers"] == new_mlp_n_layers, "No MLP layer has been added."
        if "layers_to_skip" in new_params.keys():
            assert all(
                [
                    layer_key in [f"Dense_{new_mlp_n_layers - 1}", f"Dense_{new_mlp_n_layers}"]
                    for layer_key in new_params["layers_to_skip"]
                ]
            ), "No MLP layer has been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_mlp_layer(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_mlp_layer(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_mlp_layer(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_mlp_n_layers = np.clip(
            self.hp_detail["mlp_n_layers"] - 1, *self.hp_generator.hp_space["mlp_n_layers_range"]
        )

        assert new_hp_detail["mlp_n_layers"] == new_mlp_n_layers, "No MLP layer has been added."
        if "layers_to_skip" in new_params.keys():
            assert new_params["layers_to_skip"] == [f"Dense_{new_mlp_n_layers}"], "No MLP layer has been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_mlp_layer(new_hp_detail, new_params, -1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_add_mlp_neurons(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_mlp_neurons(
            copy.deepcopy(self.hp_detail), self.params.copy(), 1
        )

        new_mlp_n_neurons = np.clip(
            self.hp_detail["mlp_n_neurons"] + 1 * 16, *self.hp_generator.hp_space["mlp_n_neurons_range"]
        )

        assert new_hp_detail["mlp_n_neurons"] == new_mlp_n_neurons, "No MLP neurons have been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_mlp_neurons(new_hp_detail, new_params, 1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_remove_mlp_neurons(self):
        print(f"-------------- Random key {self.np_key} --------------")
        new_hp_detail, new_params = self.hp_generator.add_remove_mlp_neurons(
            copy.deepcopy(self.hp_detail), self.params.copy(), -1
        )

        new_mlp_n_neurons = np.clip(
            self.hp_detail["mlp_n_neurons"] - 1 * 16, *self.hp_generator.hp_space["mlp_n_neurons_range"]
        )

        assert new_hp_detail["mlp_n_neurons"] == new_mlp_n_neurons, "No MLP neurons have been added."

        try:
            _, new_params, _ = self.hp_generator.from_hp_detail(
                jax.random.PRNGKey(self.np_key), new_hp_detail, new_params
            )
            new_hp_detail, new_params = self.hp_generator.add_remove_mlp_neurons(new_hp_detail, new_params, -1)
            self.hp_generator.from_hp_detail(jax.random.PRNGKey(self.np_key), new_hp_detail, new_params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"

    def test_multiple_transformations(self):
        key = jax.random.PRNGKey(self.np_key)
        try:
            for _ in range(30):
                transformation_key, params_key, plus_minus_key, key = jax.random.split(key, 4)

                self.hp_detail, self.params = self.hp_generator.architecture_transformations[
                    jax.random.randint(transformation_key, (), 0, 6)
                ](self.hp_detail, self.params, jax.random.choice(plus_minus_key, np.array([-1, 1])))
                _, self.params, _ = self.hp_generator.from_hp_detail(params_key, self.hp_detail, self.params)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"
