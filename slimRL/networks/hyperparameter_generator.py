from typing import Tuple, List, Callable, Dict
from functools import partial
import jax
import jax.numpy as jnp
from dehb import DEHB
from ConfigSpace import ConfigurationSpace, Integer, Float, Configuration
from slimRL.networks.single_dqn import SingleDQN


class RandomGenerator:
    def __init__(
        self,
        observation_dim,
        n_actions,
        optimizers: List[Callable],
        lr_range: Tuple,
        losses: List[Callable],
        n_layers_range: Tuple,
        n_neurons_range: Tuple,
        activations: List[Callable],
        optimizer_change_probability: float,
        architecture_change_probability: float,
    ) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions

        self.optimizers = optimizers
        self.lr_range = lr_range
        self.losses = losses
        self.n_layers_range = n_layers_range
        self.n_neurons_range = n_neurons_range
        self.activations = activations

        self.optimizer_change_probability = optimizer_change_probability
        self.architecture_change_probability = architecture_change_probability

        self.dummy_hyperparameters_fn = {
            "optimizer_hps": {"idx_optimizer": 0, "learning_rate": 0.0},
            "architecture_hps": {},
        }

    def __call__(
        self,
        key: jax.Array,
        hyperparameters_fn: Callable,
        params: Dict,
        optimizer_state: Dict,
        force_new: bool,
    ):
        key, optimizer_key, architecture_key = jax.random.split(key, 3)

        change_optimizer, hyperparameters_fn["optimizer_hps"] = self.change_optimizer_hps(
            optimizer_key, hyperparameters_fn["optimizer_hps"], force_new
        )
        if change_optimizer:
            optimizer = self.optimizers[hyperparameters_fn["optimizer_hps"]["idx_optimizer"]](
                hyperparameters_fn["optimizer_hps"]["learning_rate"]
            )
            hyperparameters_fn["optimizer_fn"] = jax.jit(optimizer.update)

            # Clean optimizer_hps
            hyperparameters_fn["optimizer_hps"] = jax.tree_map(
                lambda obj: obj.item(), hyperparameters_fn["optimizer_hps"]
            )

            # We change the architecture only if we change the optimizer
            change_architecture, idx_loss, n_layers = self.change_architecture_hps(architecture_key, force_new)

            if change_architecture:
                neurons_key, activation_key, init_key = jax.random.split(key, 3)
                hyperparameters_fn["architecture_hps"]["idx_loss"] = idx_loss
                hyperparameters_fn["architecture_hps"]["hidden_layers"] = list(
                    jax.random.randint(
                        neurons_key,
                        (n_layers,),
                        self.n_neurons_range[0],
                        self.n_neurons_range[1] + 1,
                    )
                )
                hyperparameters_fn["architecture_hps"]["indices_activations"] = list(
                    jax.random.randint(activation_key, (n_layers,), 0, len(self.activations))
                )

                q = SingleDQN(
                    self.n_actions,
                    hyperparameters_fn["architecture_hps"]["hidden_layers"],
                    [self.activations[idx] for idx in hyperparameters_fn["architecture_hps"]["indices_activations"]],
                    self.losses[hyperparameters_fn["architecture_hps"]["idx_loss"]],
                )

                hyperparameters_fn["apply_fn"] = q.apply
                hyperparameters_fn["grad_and_loss_fn"] = q.value_and_grad
                hyperparameters_fn["best_action_fn"] = q.best_action
                params = q.q_network.init(init_key, jnp.zeros(self.observation_dim, dtype=jnp.float32))

                # Clean architecture_hps
                hyperparameters_fn["architecture_hps"] = jax.tree_map(
                    lambda obj: obj.item(), hyperparameters_fn["architecture_hps"]
                )

            optimizer_state = optimizer.init(params)
        else:
            change_architecture = False

        return (
            hyperparameters_fn,
            params,
            optimizer_state,
            change_optimizer,
            change_architecture,
        )

    @partial(jax.jit, static_argnames="self")
    def change_optimizer_hps(self, key, optimizer_hps, force_new):
        change_key, generate_hp_key = jax.random.split(key)

        change_optimizer = jnp.logical_or(
            force_new,
            jax.random.bernoulli(change_key, p=self.optimizer_change_probability),
        )
        optimizer_hps = jax.lax.cond(
            change_optimizer,
            self.generate_hp_optimizer,
            lambda key, hp_: hp_,
            generate_hp_key,
            optimizer_hps,
        )

        return change_optimizer, optimizer_hps

    def generate_hp_optimizer(self, key, optimizer_hps):
        idx_key, lr_key = jax.random.split(key)

        optimizer_hps["idx_optimizer"] = jax.random.randint(idx_key, (), minval=0, maxval=len(self.optimizers))
        # sample the learning rate in log space
        optimizer_hps["learning_rate"] = 10 ** jax.random.uniform(
            lr_key, minval=self.lr_range[0], maxval=self.lr_range[1]
        )

        return optimizer_hps

    @partial(jax.jit, static_argnames="self")
    def change_architecture_hps(self, key, force_new):
        change_key, generate_hp_key = jax.random.split(key)

        change_architecture = jnp.logical_or(
            force_new,
            jax.random.bernoulli(change_key, p=self.architecture_change_probability),
        )
        idx_loss, n_layers = jax.lax.cond(
            change_architecture,
            self.generate_hp_architecture,
            lambda key: (0, 0),
            generate_hp_key,
        )

        return change_architecture, idx_loss, n_layers

    def generate_hp_architecture(self, key):
        loss_key, layers_key = jax.random.split(key)

        idx_loss = jax.random.randint(loss_key, (), minval=0, maxval=len(self.losses))
        n_layers = jax.random.randint(layers_key, (), self.n_layers_range[0], self.n_layers_range[1] + 1)

        return idx_loss, n_layers


class DEHBGenerator:
    def __init__(
        self,
        key,
        observation_dim,
        n_actions,
        optimizers: List[Callable],
        lr_range: Tuple,
        losses: List[Callable],
        n_layers_range: Tuple,
        n_neurons_range: Tuple,
        activations: List[Callable],
        min_n_epochs_per_hypeparameter,
        max_n_epochs_per_hypeparameter,
    ) -> None:
        self.observation_dim = observation_dim
        self.n_actions = n_actions

        self.optimizers = optimizers
        self.losses = losses
        self.activations = activations

        cs = ConfigurationSpace(
            seed=int(key[0]),
            space={
                "idx_optimizer": Integer("idx_optimizer", bounds=(0, len(optimizers) - 1)),
                "learning_rate": Float(
                    "learning_rate",
                    bounds=(10 ** lr_range[0], 10 ** lr_range[1]),
                    log=True,
                ),
                "idx_loss": Integer("idx_loss", bounds=(0, len(losses) - 1)),
                "n_layers": Integer("n_layers", bounds=(n_layers_range[0], n_layers_range[1])),
                "n_neurons": Integer("n_neurons", bounds=(n_neurons_range[0], n_neurons_range[1])),
                "idx_activation": Integer("idx_activation", bounds=(0, len(activations) - 1)),
            },
        )

        self.meta_optimizer = DEHB(
            cs=cs,
            dimensions=len(cs.values()),
            min_fidelity=min_n_epochs_per_hypeparameter,
            max_fidelity=max_n_epochs_per_hypeparameter,
            n_workers=1,
            output_path="experiments/dump",
        )

    def __call__(self, key, hyperparameters_fn: Dict, avg_return: float, fidelity: int):
        if hyperparameters_fn is not None:
            job_info = {
                "config": Configuration(
                    self.meta_optimizer.cs,
                    values={
                        "idx_optimizer": hyperparameters_fn["optimizer_hps"]["idx_optimizer"],
                        "learning_rate": hyperparameters_fn["optimizer_hps"]["learning_rate"],
                        "idx_loss": hyperparameters_fn["architecture_hps"]["idx_loss"],
                        "n_layers": len(hyperparameters_fn["architecture_hps"]["hidden_layers"]),
                        "n_neurons": hyperparameters_fn["architecture_hps"]["hidden_layers"][0],
                        "idx_activation": hyperparameters_fn["architecture_hps"]["indices_activations"][0],
                    },
                ),
                "fidelity": fidelity,
                "parent_id": hyperparameters_fn["parent_id"],
                "config_id": hyperparameters_fn["config_id"],
                "bracket_id": hyperparameters_fn["bracket_id"],
            }
            #  The cost of getting the fitness is egal to the requested fidelity
            self.meta_optimizer.tell(job_info, {"fitness": -avg_return, "cost": fidelity})

        # Ask for next configuration to run
        job_info = self.meta_optimizer.ask()

        hyperparameters_fn = {
            "optimizer_hps": {
                "idx_optimizer": job_info["config"]["idx_optimizer"],
                "learning_rate": job_info["config"]["learning_rate"],
            },
            "architecture_hps": {
                "idx_loss": job_info["config"]["idx_loss"],
                "hidden_layers": [job_info["config"]["n_neurons"]] * job_info["config"]["n_layers"],
                "indices_activations": [job_info["config"]["idx_activation"]] * job_info["config"]["n_layers"],
            },
            "parent_id": job_info["parent_id"],
            "config_id": job_info["config_id"],
            "bracket_id": job_info["bracket_id"],
        }

        optimizer = self.optimizers[hyperparameters_fn["optimizer_hps"]["idx_optimizer"]](
            hyperparameters_fn["optimizer_hps"]["learning_rate"]
        )
        hyperparameters_fn["optimizer_fn"] = jax.jit(optimizer.update)

        q = SingleDQN(
            self.n_actions,
            hyperparameters_fn["architecture_hps"]["hidden_layers"],
            [self.activations[idx] for idx in hyperparameters_fn["architecture_hps"]["indices_activations"]],
            self.losses[hyperparameters_fn["architecture_hps"]["idx_loss"]],
        )

        hyperparameters_fn["apply_fn"] = q.apply
        hyperparameters_fn["grad_and_loss_fn"] = q.value_and_grad
        hyperparameters_fn["best_action_fn"] = q.best_action
        params = q.q_network.init(key, jnp.zeros(self.observation_dim, dtype=jnp.float32))

        print(job_info)

        optimizer_state = optimizer.init(params)

        return hyperparameters_fn, params, optimizer_state, int(job_info["fidelity"])
