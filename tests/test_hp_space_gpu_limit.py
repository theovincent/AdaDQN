import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimdqn.networks.adadqn import AdaDQN
from slimdqn.networks import ACTIVATIONS, LOSSES, OPTIMIZERS


class TestHPSpaceGPULimit(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        hp_space = {
            "cnn_n_layers_range": [1, 3],
            "cnn_n_channels_range": [16, 64],
            "cnn_kernel_size_range": [2, 8],
            "cnn_stride_range": [2, 5],
            "mlp_n_layers_range": [0, 2],
            "mlp_n_neurons_range": [25, 512],
            "activations": list(ACTIVATIONS.values()),
            "losses": list(LOSSES.values()),
            "optimizers": list(OPTIMIZERS.values()),
            "learning_rate_range": [6, 3],
            "reset_weights": False,
        }
        self.hp_detail = {
            "cnn_n_layers": 3,
            "cnn_n_channels": 64,
            "cnn_kernel_size": 8,
            "cnn_stride": 2,
            "mlp_n_layers": 2,
            "mlp_n_neurons": 512,
            "idx_activation": 0,
            "idx_loss": 0,
            "idx_optimizer": 0,
            "learning_rate": 3,
        }
        self.np_key = np.random.randint(1000)
        self.agent = AdaDQN(
            jax.random.PRNGKey(self.np_key),
            (84, 84, 4),
            4,
            n_networks=15,
            hp_space=hp_space,
            exploitation_type="elitism",
            hp_update_frequency=100,
            gamma=0.99,
            update_horizon=1,
            update_to_data=1,
            target_update_frequency=1,
        )

        self.batch = [
            jnp.zeros((32, 84, 84, 4)),
            jnp.ones(32, dtype=jnp.int32),
            jnp.ones(32, dtype=jnp.int32),
            jnp.ones((32, 84, 84, 4)),
            None,
            None,
            jnp.zeros(32, dtype=jnp.int8),
        ]

    def test_hp_space_gpu_limit(self):
        try:
            q_key = jax.random.PRNGKey(self.np_key)
            for idx_hp in range(self.agent.n_networks):
                q_key, hp_key = jax.random.split(q_key)
                self.agent.hp_fns[idx_hp], self.agent.params[idx_hp], self.agent.optimizer_states[idx_hp] = (
                    self.agent.hp_generator.from_hp_detail(hp_key, self.hp_detail)
                )
                self.agent.hp_details[idx_hp] = self.hp_detail
                print(sum(x.size for x in jax.tree_leaves(self.agent.params[0])))

            self.agent.target_params = self.agent.params[0].copy()

            self.agent.learn_on_batch(self.batch)
            self.agent.learn_on_batch(self.batch)
        except Exception as e:
            assert 0, f"The exception {type(e).__name__} is raised. Exception: {e}"
