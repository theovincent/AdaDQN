from multiprocessing import Pool

import numpy as np
import scipy.stats


def compute_iqm_conf(array: np.ndarray, n_seeds: int, n_bootstraps: int = 2000):
    # array: n_games x n_seeds or n_seeds
    iqm = scipy.stats.trim_mean(np.sort(array.flatten()), proportiontocut=0.25)
    array_ = np.atleast_2d(array)

    bootstrap_iqms = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        values = np.array([np.random.choice(game_array, size=n_seeds) for game_array in array_])
        bootstrap_iqms[i] = scipy.stats.trim_mean(np.sort(values.flatten()), proportiontocut=0.25)

    confs = np.percentile(bootstrap_iqms, [2.5, 97.5])

    return iqm, confs


def get_iqm_and_conf_per_epoch(array: np.ndarray, n_bootstraps: int = 2000):
    # array: n_games x n_seeds x n_epochs or n_seeds x n_epochs
    n_seeds, n_epochs = array.shape[-2:]
    if array.ndim == 2 and n_seeds == 1:
        return array.reshape(-1), np.stack([array.reshape(-1), array.reshape(-1)])

    with Pool(n_epochs) as pool:
        results = pool.starmap(
            compute_iqm_conf,
            [(array[..., epoch], n_seeds, n_bootstraps) for epoch in range(n_epochs)],
        )

    iqms, confs = zip(*results)

    return np.array(iqms), np.stack(confs, axis=1)
