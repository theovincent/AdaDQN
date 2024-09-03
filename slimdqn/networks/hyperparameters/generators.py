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
        if not (self.hp_space["cnn_n_layers_range"][0] == self.hp_space["cnn_n_layers_range"][1] == 0):
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

    def from_hp_detail(self, key, hp_detail: Dict, new_params=None):
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
        if new_params is not None:
            params = jax.tree.map(
                lambda new_weights, random_weights: jnp.where(new_weights == 0, random_weights, new_weights),
                new_params,
                params,
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

    def explore(self, key, indices_new_hps, indices_replacing_hps, hp_fns, params, optimizer_states, hp_details):
        for idx in range(len(indices_new_hps)):
            new_hp_detail = hp_details[indices_replacing_hps[idx]].copy()
            new_params = copy.deepcopy(params[indices_replacing_hps[idx]])

            key, explore_key, change_key = jax.random.split(key, 3)
            random_uniform = jax.random.uniform(explore_key)

            if random_uniform < 0.2:
                key, plus_minus_key = jax.random.split(key)
                plus_minus = jax.random.choice(plus_minus_key, np.array([-1, 1]))
                if jax.random.uniform(change_key) < 0.2:
                    new_hp_detail["mlp_n_layers"] = np.clip(
                        hp_details[indices_replacing_hps[idx]]["mlp_n_layers"] + plus_minus,
                        self.config_space["mlp_n_layers"].lower,
                        self.config_space["mlp_n_layers"].upper,
                    )
                    if new_hp_detail["mlp_n_layers"] < hp_details[indices_replacing_hps[idx]]["mlp_n_layers"]:
                        layers_key = list(new_params["params"].keys())
                        new_params["params"][layers_key[-2]] = jax.tree.map(
                            lambda a: jnp.zeros_like(a), new_params["params"].pop(layers_key[-1])
                        )
                    elif new_hp_detail["mlp_n_layers"] > hp_details[indices_replacing_hps[idx]]["mlp_n_layers"]:
                        layers_key = list(new_params["params"].keys())
                        last_key = layers_key[-1]
                        new_layers_key = last_key.replace(last_key.split("_")[1], str(int(last_key.split("_")[1]) + 1))
                        new_params["params"][new_layers_key] = jax.tree.map(
                            lambda a: jnp.zeros_like(a), new_params["params"][layers_key[-1]]
                        )
                        mlp_n_neurons = hp_details[indices_replacing_hps[idx]]["mlp_n_neurons"]
                        new_params["params"][layers_key[-1]] = jax.tree.map(
                            lambda a: jnp.zeros((mlp_n_neurons,) * a.ndim), new_params["params"][layers_key[-2]]
                        )
                else:
                    new_hp_detail["mlp_n_neurons"] = np.clip(
                        hp_details[indices_replacing_hps[idx]]["mlp_n_neurons"] + plus_minus * 16,
                        self.config_space["mlp_n_neurons"].lower,
                        self.config_space["mlp_n_neurons"].upper,
                    )
                    new_mlp_n_neurons = new_hp_detail["mlp_n_neurons"]
                    mlp_n_neurons = hp_details[indices_replacing_hps[idx]]["mlp_n_neurons"]

                    if new_mlp_n_neurons < mlp_n_neurons:

                        def remove_neurons(weights: jnp.ndarray):
                            if weights.shape == (mlp_n_neurons, mlp_n_neurons):
                                return weights[:new_mlp_n_neurons, :new_mlp_n_neurons]
                            elif weights.shape == (mlp_n_neurons,) or (
                                weights.ndim == 2 and weights.shape[0] == mlp_n_neurons
                            ):
                                return weights[:new_mlp_n_neurons]
                            elif weights.ndim == 2 and weights.shape[1] == mlp_n_neurons:
                                return weights[:, :new_mlp_n_neurons]
                            else:
                                return weights

                        new_params = jax.tree.map(remove_neurons, new_params)
                    elif new_mlp_n_neurons > mlp_n_neurons:

                        def add_neurons(weights: jnp.ndarray):
                            if weights.shape == (mlp_n_neurons, mlp_n_neurons):
                                new_weights = jnp.zeros((new_mlp_n_neurons, new_mlp_n_neurons))
                                return new_weights.at[:mlp_n_neurons, :mlp_n_neurons].set(weights)
                            elif weights.ndim == 2 and weights.shape[0] == mlp_n_neurons:
                                new_weights = jnp.zeros((new_mlp_n_neurons, weights.shape[1]))
                                return new_weights.at[:mlp_n_neurons, :].set(weights)
                            elif weights.ndim == 2 and weights.shape[1] == mlp_n_neurons:
                                new_weights = jnp.zeros((weights.shape[0], new_mlp_n_neurons))
                                return new_weights.at[:, :mlp_n_neurons].set(weights)
                            elif weights.shape == (mlp_n_neurons,):
                                new_weights = jnp.zeros(new_mlp_n_neurons)
                                return new_weights.at[:mlp_n_neurons].set(weights)
                            else:
                                return weights

                        new_params = jax.tree.map(add_neurons, new_params)

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
                self.from_hp_detail(hp_key, new_hp_detail, new_params)
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
            if not (self.hp_space["cnn_n_layers_range"][0] == self.hp_space["cnn_n_layers_range"][1] == 0):
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
        if not (self.hp_space["cnn_n_layers_range"][0] == self.hp_space["cnn_n_layers_range"][1] == 0):
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
