import os
import json
import pickle
import time

SHARED_PARAMS = [
    "experiment_name",
    "env",
    "replay_capacity",
    "batch_size",
    "update_horizon",
    "gamma",
    "horizon",
    "update_to_data",
    "target_update_frequency",
    "n_initial_samples",
    "end_epsilon",
    "duration_epsilon",
    "n_epochs",
    "n_training_steps_per_epoch",
]

AGENT_PARAMS = {
    "DQN": ["hidden_layers", "activation", "lr", "optimizer", "loss"],
    "AdaDQN": [
        "n_networks",
        "n_layers_range",
        "n_neurons_range",
        "activations",
        "lr_range",
        "optimizers",
        "losses",
        "end_online_exp",
    ],
    "RSDQN": ["n_layers_range", "n_neurons_range", "activations", "lr_range", "optimizers", "losses"],
}


def check_experiment(p: dict):
    # check if the experiment is valid
    returns_path = os.path.join(p["save_path"], "returns_seed_" + str(p["seed"]) + ".npy")
    losses_path = os.path.join(p["save_path"], "losses_seed_" + str(p["seed"]) + ".npy")
    model_path = os.path.join(p["save_path"], "model_seed_" + str(p["seed"]))

    assert not (
        os.path.exists(returns_path) or os.path.exists(losses_path) or os.path.exists(model_path)
    ), "Same algorithm with same seed results already exists. Delete them and restart, or change the experiment name."

    params_path = os.path.join(
        os.path.split(p["save_path"])[0],  # parameters.json is outside the algorithm folder (in the experiment folder)
        "parameters.json",
    )

    if os.path.exists(params_path):
        # when many seed are launched at the same time, the params exist but they are still being dumped
        try:
            params = json.load(open(params_path, "r"))
            for param in SHARED_PARAMS:
                assert (
                    params[param] == p[param]
                ), "Same experiment has been run with different shared parameters. Change the experiment name."
            if f"---- {p['algo']} ---" in params.keys():
                for param in AGENT_PARAMS[p["algo"]]:
                    assert (
                        params[param] == p[param]
                    ), f"Same experiment has been run with different {p['algo']} parameters. Change the experiment name."
        except json.decoder.JSONDecodeError:
            pass
    else:
        # if the folder exists for a long time then raise an error
        if (
            os.path.exists(os.path.join(p["save_path"], ".."))
            and (time.time() - os.path.getmtime(os.path.join(p["save_path"], ".."))) > 4
        ):
            assert (
                True
            ), "There is a folder with this experiment name and no parameters.json. Delete the folder and restart, or change the experiment name."


def store_params(p: dict):
    params_path = os.path.join(
        p["save_path"],
        "..",
        "parameters.json",
    )

    if os.path.exists(params_path):
        # when many seed are launched at the same time, the params exist but they are still being dumped
        loaded = False
        while not loaded:
            try:
                params = json.load(open(params_path, "r"))
                loaded = True
            except json.decoder.JSONDecodeError:
                pass
    else:
        params = {}

        # store shared params
        params["---- Shared parameters ---"] = "----------------"
        for shared_param in SHARED_PARAMS:
            params[shared_param] = p[shared_param]

    if f"---- {p['algo']} ---" not in params.keys():
        # store algo params
        params[f"---- {p['algo']} ---"] = "-----------------------------"
        for agent_param in AGENT_PARAMS[p["algo"]]:
            params[agent_param] = p[agent_param]

    # set parameter order for sorting all keys in a pre-defined order
    algo_params = []
    for agent in sorted(AGENT_PARAMS):
        if f"---- {agent} ---" in params:
            algo_params = algo_params + [f"---- {agent} ---"] + AGENT_PARAMS[agent]

    params_order = SHARED_PARAMS + algo_params

    # sort keys in uniform order and store
    params = {key: params[key] for key in params_order}

    json.dump(params, open(params_path, "w"), indent=4)


def prepare_logs(p: dict):
    check_experiment(p)
    os.makedirs(p["save_path"], exist_ok=True)  # need to create a directory for this experiment, algorithm combination
    store_params(p)


def pickle_load(path):
    return pickle.load(open(path, "rb"))


def pickle_dump(obj, path):
    return pickle.dump(obj, open(path, "wb"))


def save_logs(p: dict, log_rewards: list, log_lengths: list, model):
    rewards_path = os.path.join(p["save_path"], f"rewards_seed_{p['seed']}.json")
    lengths_path = os.path.join(p["save_path"], f"lengths_seed_{p['seed']}.json")
    model_path = os.path.join(p["save_path"], f"model_seed_{p['seed']}")

    json.dump(log_rewards, open(rewards_path, "w"), indent=4)
    json.dump(log_lengths, open(lengths_path, "w"), indent=4)
    pickle_dump(model, model_path)
