import numpy as np


def get_core_list(ncells, max_cells_per_core=30000, min_cells_per_core=2):
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
    min_graph_size = max(2, int(ncells / max_cells_per_core))
    max_graph_size = int(ncells / min_cells_per_core)

    cores = []
    for candidate in range(min_graph_size, max_graph_size):
        factors = _prime_factors(candidate)
        twos = np.count_nonzero(factors == 2)
        fives = np.count_nonzero(factors == 5)
        gt_five = np.count_nonzero(factors > 5)
        big_factor = factors.max()
        if twos > 0 and fives <= twos and gt_five <= 1 and big_factor <= 7:
            cores.append(candidate)
        # small odd multiples of 3 and a few that correspond to divisors of the
        # ne30 (30x30x6=5400) size
        elif candidate in [3, 9, 15, 21, 225, 675, 1350]:
            cores.append(candidate)

    return np.array(cores)


# https://stackoverflow.com/a/22808285
def _prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return np.array(factors)
