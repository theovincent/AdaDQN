from collections import Counter
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
                    bounds=(10 ** -hp_space["learning_rate_range"][0], 10 ** -hp_space["learning_rate_range"][1]),
                    log=True,
                ),
            },
        )

    def from_hp_detail(self, key, hp_detail):
        q = BaseDQN(
            [hp_detail["n_neurons"]] * hp_detail["n_layers"],
            [self.hp_space["activations"][hp_detail["idx_activation"]]] * hp_detail["n_layers"],
            self.hp_space["cnn"],
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
        optimizer_state = optimizer.init(params)

        return hp_fn, params, optimizer_state

    def sample(self, key):
        hp_detail = dict(self.config_space.sample_configuration())

        hp_fn, params, optimizer_state = self.from_hp_detail(key, hp_detail)

        return hp_fn, params, optimizer_state, hp_detail

    def exploit_and_explore(self, key, metrics, hp_fns, params, optimizer_states, hp_details):
        n_networks = len(metrics)
        # exploit
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

        # explore
        for idx in range(len(indices_new_hps)):
            new_hp_detail = hp_details[indices_replacing_hps[idx]].copy()

            key, explore_key, change_key = jax.random.split(key, 3)
            random_uniform = jax.random.uniform(explore_key)

            if random_uniform < 0.2:
                key, plus_minus_key = jax.random.split(key)
                plus_minus = jax.random.choice(plus_minus_key, np.array([-1, 1]))
                if jax.random.uniform(change_key) < 0.2:
                    new_hp_detail["n_layers"] = np.clip(
                        hp_details[indices_replacing_hps[idx]]["n_layers"] + plus_minus,
                        self.config_space["n_layers"].lower,
                        self.config_space["n_layers"].upper,
                    )
                else:
                    new_hp_detail["n_neurons"] = np.clip(
                        hp_details[indices_replacing_hps[idx]]["n_neurons"] + plus_minus * 16,
                        self.config_space["n_neurons"].lower,
                        self.config_space["n_neurons"].upper,
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
                self.from_hp_detail(hp_key, new_hp_detail)
            )
            hp_details[indices_new_hps[idx]] = new_hp_detail

        return indices_new_hps, hp_fns, params, optimizer_states, hp_details

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
            job_info = {
                "config": Configuration(
                    self.config_space,
                    values={
                        "n_layers": self.hp_detail["n_layers"],
                        "n_neurons": self.hp_detail["n_neurons"],
                        "idx_activation": self.hp_detail["idx_activation"],
                        "idx_loss": self.hp_detail["idx_loss"],
                        "idx_optimizer": self.hp_detail["idx_optimizer"],
                        "learning_rate": self.hp_detail["learning_rate"],
                    },
                ),
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
            "n_layers": job_info["config"]["n_layers"],
            "n_neurons": job_info["config"]["n_neurons"],
            "idx_activation": job_info["config"]["idx_activation"],
            "idx_loss": job_info["config"]["idx_loss"],
            "idx_optimizer": job_info["config"]["idx_optimizer"],
            "learning_rate": job_info["config"]["learning_rate"],
        }

        hp_fn, params, optimizer_state = self.from_hp_detail(key, self.hp_detail)

        self.hp_extra_detail = {
            "fidelity": job_info["fidelity"],
            "parent_id": job_info["parent_id"],
            "config_id": job_info["config_id"],
            "bracket_id": job_info["bracket_id"],
        }

        return hp_fn, params, optimizer_state, self.hp_detail, int(job_info["fidelity"])
