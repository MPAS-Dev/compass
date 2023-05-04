import xarray as xr


def get_ntasks_from_cell_count(config, at_setup, mesh_filename):
    """
    Computes ``ntasks`` and ``min_tasks`` for a step based on the estimated
    (at setup) or exact (at runtime) mesh size

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case the step belongs to

    at_setup : bool
        Whether this method is being run during setup of the step, as
        opposed to at runtime

    mesh_filename : str
        A file containing the MPAS mesh (specifically ``nCells``) if
        ``at_setup == False``

    Returns
    -------
    ntasks : int
        The target number of MPI tasks for the step

    min_tasks : int
        The minimum number of tasks for the step
    """
    if at_setup:
        cell_count = config.getint('global_ocean', 'approx_cell_count')
    else:
        with xr.open_dataset(mesh_filename) as ds:
            cell_count = ds.sizes['nCells']

    if cell_count is None:
        raise ValueError('ntasks and min_tasks were not set explicitly '
                         'but they also cannot be computed because '
                         'compute_cell_count() does not appear to have '
                         'been overridden.')

    goal_cells_per_core = config.getfloat('global_ocean',
                                          'goal_cells_per_core')
    max_cells_per_core = config.getfloat('global_ocean',
                                         'max_cells_per_core')

    # machines (e.g. Perlmutter) seem to be happier with ntasks that
    # are multiples of 4
    # ideally, about 200 cells per core
    ntasks = max(1, 4 * round(cell_count / (4 * goal_cells_per_core)))
    # In a pinch, about 2000 cells per core
    min_tasks = max(1, 4 * round(cell_count / (4 * max_cells_per_core)))

    return ntasks, min_tasks
