import multiprocessing


def calculate_decomp_core_pair(config, target_max_tasks,
                               smallest_acceptable_max_tasks):
    """
    Get a pair of decomposition task counts based on available cores.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    target_max_tasks : int
        Preferred upper bound on the larger decomposition

    smallest_acceptable_max_tasks : int
        Minimum required size of the larger decomposition

    Returns
    -------
    proc_list : list of int
        The pair of decomposition task counts, ordered as
        ``[low_tasks, max_tasks]``
    """

    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        if config.has_option('parallel', 'cores_per_node'):
            cores_per_node = config.getint('parallel',
                                           'cores_per_node')
        else:
            raise ValueError('Expected parallel system slurm to have '
                             'option cores_per_node')
    elif parallel_system == 'single_node':
        cores_per_node = multiprocessing.cpu_count()
        cores_per_node = min(cores_per_node,
                             config.getint('parallel', 'cores_per_node'))
    else:
        raise ValueError(f'Unexpected parallel system {parallel_system}')

    max_tasks = max(smallest_acceptable_max_tasks,
                    min(target_max_tasks, cores_per_node))
    low_tasks = max(1, max_tasks // 2)
    return [low_tasks, max_tasks]
