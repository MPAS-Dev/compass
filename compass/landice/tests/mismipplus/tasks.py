import xarray as xr

# Following from:
#   compass/ocean/tests/global_ocean/tasks.py


def get_ntasks_from_cell_count(config, at_setup, mesh_filename):
    """
    Computes `ntasks` and `min_tasks` based on mesh size (i.e. nCells)

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case the step belongs to

    at_setup: bool
        Whether this method is being run during setup of the step, as
        opposed to at runtime

    mesh_filename : str
        A file containing the MPAS mesh (specifically `nCells`) if
        `at_setup == False`

    Returns
    -------
    ntasks : int
        The target number of MPI tasks

    min_tasks : int
        The minimum number of tasks
    """

    if at_setup:
        # get the desired resolution from config file
        resolution = config.getfloat("mesh", "resolution")
        # get the reference number of cells as 1000m resolution
        ncells_at_1km_res = config.getfloat("mismipplus", "ncells_at_1km_res")
        # approximate the number of cells for the desired resolution
        cell_count = int(ncells_at_1km_res * (1000 / resolution)**2)
        # TODO: Account for extra cells when gutter is requested
    else:
        # get cell count from mesh
        with xr.open_dataset(mesh_filename) as ds:
            cell_count = ds.sizes['nCells']

    # read MPI related parameters from configuration file
    cores_per_node = config.getfloat('parallel', 'cores_per_node')
    max_cells_per_core = config.getfloat('mismipplus', 'max_cells_per_core')
    goal_cells_per_core = config.getfloat('mismipplus', 'goal_cells_per_core')

    # from Xylar: machines (e.g. Perlmutter) seem to be happier with ntasks
    #             that are multiples of 4
    min_tasks = max(1, 4 * round(cell_count / (4 * max_cells_per_core)))
    ntasks = max(1, 4 * round(cell_count / (4 * goal_cells_per_core)))
    # if ntasks exceeds the number of cores per node, round to the nearest
    # multiple of `cores_per_node`.
    if ntasks > cores_per_node:
        ntasks = int(cores_per_node) * round(ntasks / cores_per_node)

    return ntasks, min_tasks


def _approx_cell_count(config):
    """
    Approximate the number of cells based on the resolution ratio squared
    times the number of cells at 1km resolution. Also do a crude area scaling
    to account for the additional cells from the gutter (if present).
    """
    # Fixed domain lenghts [m] (without gutter)
    Lx = 640e3
    Ly = 80e3
    # Reference domain area [m^2]
    ref_area = Lx * Ly

    # get the desired resolution from config file
    resolution = config.getfloat("mesh", "resolution")
    # get the requested gutter length
    gutterLength = config.getfloat("mesh", "gutter_length")
    # ensure that the requested `gutterLength` is valid. Otherwise set
    # the value to zero, such that the default `gutterLength` of two
    # gridcells is used.
    if (gutterLength < 2. * resolution) and (gutterLength != 0.):
        gutterLength = 0.

    # get the reference number of cells as 1000m resolution
    ncells_at_1km_res = config.getfloat("mismipplus", "ncells_at_1km_res")
    # approximate the number of cells for the desired resolution
    cell_count = int(ncells_at_1km_res * (1000 / resolution)**2)

    # Account for extra cells when gutter is requested
    if gutterLength != 0.:
        # calculate the area of the domain with the gutter
        new_area = (Lx + gutterLength) * Ly
        # scale by the approx cell count by the relative area increase
        # due to presence of the gutter
        cell_count *= ref_area / new_area
