import numpy as np


def get_core_list(ncells, max_cells_per_core=6000, min_cells_per_core=100):
    """
    Get a fairly exhaustive list of core counts to partition a given number of
    cells into

    Parameters
    ----------
    ncells : int
        The number of cells in the mesh

    max_cells_per_core : float, optional
        the approximate maximum number of cells per core (use do determine
        the minimum number of cores allowed)

    min_cells_per_core : float, optional
        the approximate minimum number of cells per core (use do determine
        the maximum number of cores allowed)

    Returns
    -------
    cores : numpy.ndarray
        Likely numbers of cores to run with
    """
    min_graph_size = int(ncells / max_cells_per_core)
    max_graph_size = int(ncells / min_cells_per_core)
    n_power2 = 2**np.arange(1, 21)
    n_multiples12 = 12 * np.arange(1, 9)

    cores = n_power2
    for power10 in range(3):
        cores = np.concatenate([cores, 10**power10 * n_multiples12])

    mask = np.logical_and(cores >= min_graph_size,
                          cores <= max_graph_size)
    cores = cores[mask]

    return cores
