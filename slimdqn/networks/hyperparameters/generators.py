from typing import Dict
from collections import Counter
import copy
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
        self.cnn = not (self.hp_space["cnn_n_layers_range"][0] == self.hp_space["cnn_n_layers_range"][1] == 0)

        space = {
            "mlp_n_layers": Integer(
                "mlp_n_layers", bounds=(hp_space["mlp_n_layers_range"][0], hp_space["mlp_n_layers_range"][1])
            ),
            "mlp_n_neurons": Integer(
                "mlp_n_neurons", bounds=(hp_space["mlp_n_neurons_range"][0], hp_space["mlp_n_neurons_range"][1])
            ),
            "idx_activation": Integer("idx_activation", bounds=(0, len(hp_space["activations"]) - 1)),
            "idx_loss": Integer("idx_loss", bounds=(0, len(hp_space["losses"]) - 1)),
            "idx_optimizer": Integer("idx_optimizer", bounds=(0, len(hp_space["optimizers"]) - 1)),
            "learning_rate": Float(
                "learning_rate",
                bounds=(10 ** -hp_space["learning_rate_range"][0], 10 ** -hp_space["learning_rate_range"][1]),
                log=True,
            ),
        }
        if self.cnn:
            space["cnn_n_layers"] = Integer(
                "cnn_n_layers", bounds=(hp_space["cnn_n_layers_range"][0], hp_space["cnn_n_layers_range"][1])
            )
            space["cnn_n_channels"] = Integer(
                "cnn_n_channels", bounds=(hp_space["cnn_n_channels_range"][0], hp_space["cnn_n_channels_range"][1])
            )
            space["cnn_kernel_size"] = Integer(
                "cnn_kernel_size", bounds=(hp_space["cnn_kernel_size_range"][0], hp_space["cnn_kernel_size_range"][1])
            )
            space["cnn_stride"] = Integer(
                "cnn_stride", bounds=(hp_space["cnn_stride_range"][0], hp_space["cnn_stride_range"][1])
            )

        self.config_space = ConfigurationSpace(seed=int(key[1]), space=space)

        self.architecture_transformations = [
            self.add_remove_cnn_layer,
            self.add_remove_cnn_channels,
            self.add_remove_cnn_kernel_size,
            self.add_remove_cnn_stride,
            self.add_remove_mlp_layer,
            self.add_remove_mlp_neurons,
        ]

    def from_hp_detail(self, key, hp_detail: Dict, old_params=None):
        q = BaseDQN(
            hp_detail.get("cnn_n_layers", 0),
            hp_detail.get("cnn_n_channels", 0),
            hp_detail.get("cnn_kernel_size", 0),
            hp_detail.get("cnn_stride", 0),
            hp_detail["mlp_n_layers"],
            hp_detail["mlp_n_neurons"],
            self.hp_space["activations"][hp_detail["idx_activation"]],
            self.n_actions,
            self.hp_space["losses"][hp_detail["idx_loss"]],
        )
        optimizer = self.hp_space["optimizers"][hp_detail["idx_optimizer"]](hp_detail["learning_rate"])

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
        params = q.q_network.init(key, jnp.zeros(self.observation_dim))
        if old_params is not None:
            for key_layer in params["params"]:
                if key_layer not in old_params.get("layers_to_skip", []):
                    params["params"][key_layer] = jax.tree.map(
                        lambda old_weights, random_weights: jnp.where(old_weights == 0, random_weights, old_weights),
                        old_params["params"][key_layer],
                        params["params"][key_layer],
                    )

        optimizer_state = optimizer.init(params)

        return hp_fn, params, optimizer_state

    def sample(self, key):
        hp_detail = dict(self.config_space.sample_configuration())

        hp_fn, params, optimizer_state = self.from_hp_detail(key, hp_detail)

        return hp_fn, params, optimizer_state, hp_detail

    def exploit(self, key, metrics):
        n_networks = len(metrics)

        if self.exploitation_type == "elitism":
            # Make sure the best HP is kept
            selected_indices = [np.nanargmax(metrics)]
            for _ in range(n_networks - 1):
                key, selection_key = jax.random.split(key)
                random_indices = jax.random.choice(selection_key, jnp.arange(n_networks), (3,), replace=False)
                selected_indices.append(random_indices[np.nanargmax(metrics[random_indices])].item())

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
        elif self.exploitation_type == "truncation":
            cut_new_hps = np.around(n_networks * 0.2).astype(int)
            cut_replacing_hps = n_networks - cut_new_hps
            partition_indices_ = np.argpartition(metrics, (cut_new_hps, cut_replacing_hps))
            # Replace the nans first
            partition_indices = np.roll(partition_indices_, np.isnan(metrics).sum())

            indices_new_hps = partition_indices[:cut_new_hps]
            indices_replacing_hps = partition_indices[cut_replacing_hps:]

        return key, indices_new_hps, indices_replacing_hps

    def add_remove_cnn_layer(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_cnn_n_layers = hp_detail["cnn_n_layers"]
        hp_detail["cnn_n_layers"] = np.clip(
            hp_detail["cnn_n_layers"] + plus_minus,
            self.config_space["cnn_n_layers"].lower,
            self.config_space["cnn_n_layers"].upper,
        )
        cnn_layers_key = list(params["params"].keys())[: hp_detail["cnn_n_layers"]]
        if hp_detail["cnn_n_layers"] < old_cnn_n_layers:
            params["params"].pop(cnn_layers_key[-1])
            params["params"]["layers_to_skip"] = [cnn_layers_key[-2]]
        elif hp_detail["cnn_n_layers"] > old_cnn_n_layers:
            last_key = cnn_layers_key[-1]
            last_layer_number = last_key.split("_")[1]
            new_layers_key = last_key.replace(last_layer_number, str(int(last_layer_number) + 1))
            params["params"]["layers_to_skip"] = [last_key, new_layers_key]

        if hp_detail["cnn_n_layers"] != old_cnn_n_layers:
            params["params"]["layers_to_skip"].extend(list(params["params"].keys())[hp_detail["cnn_n_layers"] :])

        return hp_detail, params

    def add_remove_cnn_channels(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_cnn_n_channels = hp_detail["cnn_n_channels"]
        hp_detail["cnn_n_channels"] = np.clip(
            hp_detail["cnn_n_channels"] + 4 * plus_minus,
            self.config_space["cnn_n_channels"].lower,
            self.config_space["cnn_n_channels"].upper,
        )

        def remove_channels(weights: jnp.ndarray):
            # first CNN layer
            if weights.ndim == 4 and weights.shape[2] != old_cnn_n_channels:
                return weights[:, :, :, : hp_detail["cnn_n_channels"]]
            elif weights.ndim == 4 and weights.shape[2] == old_cnn_n_channels:
                return weights[:, :, : hp_detail["cnn_n_channels"], : hp_detail["cnn_n_channels"]]
            else:
                return weights[: hp_detail["cnn_n_channels"]]

        def add_channels(weights: jnp.ndarray):
            # first CNN layer
            if weights.ndim == 4 and weights.shape[2] != old_cnn_n_channels:
                new_weights = jnp.zeros(
                    (weights.shape[0], weights.shape[1], weights.shape[2], hp_detail["cnn_kernel_size"])
                )
                return new_weights.at[:, :, :, :old_cnn_n_channels].set(weights)
            elif weights.ndim == 4 and weights.shape[2] == old_cnn_n_channels:
                new_weights = jnp.zeros(
                    (weights.shape[0], weights.shape[1], hp_detail["cnn_kernel_size"], hp_detail["cnn_kernel_size"])
                )
                return new_weights.at[:, :, :old_cnn_n_channels, :old_cnn_n_channels].set(weights)
            else:
                return jnp.zeros(weights.shape[0]).at[:old_cnn_n_channels].set(weights)

        if hp_detail["cnn_kernel_size"] < old_cnn_n_channels:
            for cnn_key in list(list(params["params"].keys())[: hp_detail["cnn_n_layers"]]):
                params["params"][cnn_key] = jax.tree.map(remove_channels, params["params"][cnn_key])
        elif hp_detail["cnn_kernel_size"] > old_cnn_n_channels:
            for cnn_key in list(list(params["params"].keys())[: hp_detail["cnn_n_layers"]]):
                params["params"][cnn_key] = jax.tree.map(add_channels, params["params"][cnn_key])

        if hp_detail["cnn_n_channels"] != old_cnn_n_channels:
            params["params"]["layers_to_skip"] = list(params["params"].keys())[hp_detail["cnn_n_layers"] :]

        return hp_detail, params

    def add_remove_cnn_kernel_size(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_cnn_kernel_size = hp_detail["cnn_kernel_size"]
        hp_detail["cnn_kernel_size"] = np.clip(
            hp_detail["cnn_kernel_size"] + plus_minus,
            self.config_space["cnn_kernel_size"].lower,
            self.config_space["cnn_kernel_size"].upper,
        )

        def reduce_kernel_size(weights: jnp.ndarray):
            if weights.ndim == 4:
                return weights[: hp_detail["cnn_kernel_size"], : hp_detail["cnn_kernel_size"]]
            else:
                return weights

        def increase_kernel_size(weights: jnp.ndarray):
            if weights.ndim == 4:
                new_weights = jnp.zeros(
                    (hp_detail["cnn_kernel_size"], hp_detail["cnn_kernel_size"], weights.shape[2], weights.shape[3])
                )
                return new_weights.at[:old_cnn_kernel_size, :old_cnn_kernel_size].set(weights)
            else:
                return weights

        if hp_detail["cnn_kernel_size"] < old_cnn_kernel_size:
            for cnn_key in list(list(params["params"].keys())[: hp_detail["cnn_n_layers"]]):
                params["params"][cnn_key] = jax.tree.map(reduce_kernel_size, params["params"][cnn_key])
        elif hp_detail["cnn_kernel_size"] > old_cnn_kernel_size:
            for cnn_key in list(list(params["params"].keys())[: hp_detail["cnn_n_layers"]]):
                params["params"][cnn_key] = jax.tree.map(increase_kernel_size, params["params"][cnn_key])

        if hp_detail["cnn_kernel_size"] != old_cnn_kernel_size:
            params["params"]["layers_to_skip"] = list(params["params"].keys())[hp_detail["cnn_n_layers"] :]

        return hp_detail, params

    def add_remove_cnn_stride(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_cnn_stride = hp_detail["cnn_stride"]
        hp_detail["cnn_stride"] = np.clip(
            hp_detail["cnn_stride"] + plus_minus,
            self.config_space["cnn_stride"].lower,
            self.config_space["cnn_stride"].upper,
        )

        if hp_detail["cnn_stride"] != old_cnn_stride:
            params["params"]["layers_to_skip"] = list(params["params"].keys())[hp_detail["cnn_n_layers"] :]

        return hp_detail, params

    def add_remove_mlp_layer(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_mlp_n_layers = hp_detail["mlp_n_layers"]
        hp_detail["mlp_n_layers"] = np.clip(
            hp_detail["mlp_n_layers"] + plus_minus,
            self.config_space["mlp_n_layers"].lower,
            self.config_space["mlp_n_layers"].upper,
        )
        layers_key = list(params["params"].keys())
        if hp_detail["mlp_n_layers"] < old_mlp_n_layers:
            params["params"].pop(layers_key[-1])
            params["params"]["layers_to_skip"] = [layers_key[-2]]
        elif hp_detail["mlp_n_layers"] > old_mlp_n_layers:
            last_key = layers_key[-1]
            last_layer_number = last_key.split("_")[1]
            new_layers_key = last_key.replace(last_layer_number, str(int(last_layer_number) + 1))
            params["params"]["layers_to_skip"] = [last_key, new_layers_key]

        return hp_detail, params

    def add_remove_mlp_neurons(self, hp_detail: Dict, params: Dict, plus_minus: int):
        old_mlp_n_neurons = hp_detail["mlp_n_neurons"]
        hp_detail["mlp_n_neurons"] = np.clip(
            hp_detail["mlp_n_neurons"] + plus_minus * 16,
            self.config_space["mlp_n_neurons"].lower,
            self.config_space["mlp_n_neurons"].upper,
        )

        def remove_neurons(weights: jnp.ndarray):
            if weights.shape == (old_mlp_n_neurons, old_mlp_n_neurons):
                return weights[: hp_detail["mlp_n_neurons"], : hp_detail["mlp_n_neurons"]]
            elif weights.shape == (old_mlp_n_neurons,) or (weights.ndim == 2 and weights.shape[0] == old_mlp_n_neurons):
                return weights[: hp_detail["mlp_n_neurons"]]
            elif weights.ndim == 2 and weights.shape[1] == old_mlp_n_neurons:
                return weights[:, : hp_detail["mlp_n_neurons"]]
            else:
                return weights

        def add_neurons(weights: jnp.ndarray):
            if weights.shape == (old_mlp_n_neurons, old_mlp_n_neurons):
                new_weights = jnp.zeros((hp_detail["mlp_n_neurons"], hp_detail["mlp_n_neurons"]))
                return new_weights.at[:old_mlp_n_neurons, :old_mlp_n_neurons].set(weights)
            elif weights.ndim == 2 and weights.shape[0] == old_mlp_n_neurons:
                new_weights = jnp.zeros((hp_detail["mlp_n_neurons"], weights.shape[1]))
                return new_weights.at[:old_mlp_n_neurons, :].set(weights)
            elif weights.ndim == 2 and weights.shape[1] == old_mlp_n_neurons:
                new_weights = jnp.zeros((weights.shape[0], hp_detail["mlp_n_neurons"]))
                return new_weights.at[:, :old_mlp_n_neurons].set(weights)
            elif weights.shape == (old_mlp_n_neurons,):
                return jnp.zeros(hp_detail["mlp_n_neurons"]).at[:old_mlp_n_neurons].set(weights)
            else:
                return weights

        if hp_detail["mlp_n_neurons"] < old_mlp_n_neurons:
            for mlp_key in list(list(params["params"].keys())[-hp_detail["mlp_n_layers"] :]):
                params["params"][mlp_key] = jax.tree.map(remove_neurons, params["params"][mlp_key])
        elif hp_detail["mlp_n_neurons"] > old_mlp_n_neurons:
            for mlp_key in list(list(params["params"].keys())[-hp_detail["mlp_n_layers"] :]):
                params["params"][mlp_key] = jax.tree.map(add_neurons, params["params"][mlp_key])

        return hp_detail, params

    def explore(self, key, indices_new_hps, indices_replacing_hps, hp_fns, params, optimizer_states, hp_details):
        for idx in range(len(indices_new_hps)):
            new_hp_detail = hp_details[indices_replacing_hps[idx]].copy()
            old_params = copy.deepcopy(params[indices_replacing_hps[idx]])

            key, explore_key, change_key = jax.random.split(key, 3)
            random_uniform = jax.random.uniform(explore_key)

            if random_uniform < 0.2:
                key, plus_minus_key, transformation_key = jax.random.split(key, 3)
                plus_minus = jax.random.choice(plus_minus_key, np.array([-1, 1]))
                idx_transformation = jax.random.choice(
                    transformation_key,
                    jnp.arange(6),
                    p=(
                        np.array([0.1, 0.4 / 3, 0.4 / 3, 0.4 / 3, 0.1, 0.4])
                        if self.cnn
                        else np.array([0, 0, 0, 0, 0.2, 0.8])
                    ),
                )
                new_hp_detail, old_params = self.architecture_transformations[idx_transformation](
                    new_hp_detail, old_params, plus_minus
                )

            elif random_uniform < 0.4:
                new_hp_detail["idx_activation"] = jax.random.randint(
                    change_key, (), 0, self.config_space["idx_activation"].upper, dtype=int
                )
            elif random_uniform < 0.6:
                new_hp_detail["idx_loss"] = jax.random.randint(
                    change_key, (), 0, self.config_space["idx_loss"].upper, dtype=int
                )
            elif random_uniform < 0.8:
                new_hp_detail["idx_optimizer"] = jax.random.randint(
                    change_key, (), 0, self.config_space["idx_optimizer"].upper, dtype=int
                )
            else:
                key, plus_minus_key = jax.random.split(key)
                plus_minus = jax.random.choice(plus_minus_key, np.array([-1, 1]))
                new_hp_detail["learning_rate"] = np.clip(
                    hp_details[indices_replacing_hps[idx]]["learning_rate"] * (1 + plus_minus * 0.2),
                    self.config_space["learning_rate"].lower,
                    self.config_space["learning_rate"].upper,
                )

            key, hp_key = jax.random.split(key)
            hp_fns[indices_new_hps[idx]], params[indices_new_hps[idx]], optimizer_states[indices_new_hps[idx]] = (
                self.from_hp_detail(hp_key, new_hp_detail, old_params)
            )
            hp_details[indices_new_hps[idx]] = new_hp_detail

        return indices_new_hps, hp_fns, params, optimizer_states, hp_details

    def exploit_and_explore(self, key, metrics, hp_fns, params, optimizer_states, hp_details):
        explore_key, indices_new_hps, indices_replacing_hps = self.exploit(key, metrics)
        return self.explore(
            explore_key, indices_new_hps, indices_replacing_hps, hp_fns, params, optimizer_states, hp_details
        )

    def dehb_init(self, min_n_epochs_per_hp, max_n_epochs_per_hp):
        self.meta_optimizer = DEHB(
            cs=self.config_space,
            dimensions=len(self.config_space.values()),
            min_fidelity=min_n_epochs_per_hp,
            max_fidelity=max_n_epochs_per_hp,
            n_workers=1,
            output_path="experiments/dump",
        )

    def dehb_sample(self, key, avg_return):
        if avg_return is not None:
            values = {
                "mlp_n_layers": self.hp_detail["mlp_n_layers"],
                "mlp_n_neurons": self.hp_detail["mlp_n_neurons"],
                "idx_activation": self.hp_detail["idx_activation"],
                "idx_loss": self.hp_detail["idx_loss"],
                "idx_optimizer": self.hp_detail["idx_optimizer"],
                "learning_rate": self.hp_detail["learning_rate"],
            }
            if self.cnn:
                values["cnn_n_layers"] = self.hp_detail["cnn_n_layers"]
                values["cnn_n_channels"] = self.hp_detail["cnn_n_channels"]
                values["cnn_kernel_size"] = self.hp_detail["cnn_kernel_size"]
                values["cnn_stride"] = self.hp_detail["cnn_stride"]
            job_info = {
                "config": Configuration(self.config_space, values=values),
                "fidelity": self.hp_extra_detail["fidelity"],
                "parent_id": self.hp_extra_detail["parent_id"],
                "config_id": self.hp_extra_detail["config_id"],
                "bracket_id": self.hp_extra_detail["bracket_id"],
            }
            #  The cost of getting the fitness is egal to the requested fidelity
            self.meta_optimizer.tell(job_info, {"fitness": -avg_return, "cost": int(job_info["fidelity"])})

        # Ask for next configuration to run
        job_info = self.meta_optimizer.ask()

        self.hp_detail = {
            "mlp_n_layers": job_info["config"]["mlp_n_layers"],
            "mlp_n_neurons": job_info["config"]["mlp_n_neurons"],
            "idx_activation": job_info["config"]["idx_activation"],
            "idx_loss": job_info["config"]["idx_loss"],
            "idx_optimizer": job_info["config"]["idx_optimizer"],
            "learning_rate": job_info["config"]["learning_rate"],
        }
        if self.cnn:
            self.hp_detail["cnn_n_layers"] = job_info["config"]["cnn_n_layers"]
            self.hp_detail["cnn_n_channels"] = job_info["config"]["cnn_n_channels"]
            self.hp_detail["cnn_kernel_size"] = job_info["config"]["cnn_kernel_size"]
            self.hp_detail["cnn_stride"] = job_info["config"]["cnn_stride"]

        hp_fn, params, optimizer_state = self.from_hp_detail(key, self.hp_detail)

        self.hp_extra_detail = {
            "fidelity": job_info["fidelity"],
            "parent_id": job_info["parent_id"],
            "config_id": job_info["config_id"],
            "bracket_id": job_info["bracket_id"],
        }

        return hp_fn, params, optimizer_state, self.hp_detail, int(job_info["fidelity"])
