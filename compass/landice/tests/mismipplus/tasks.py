import xarray as xr

# Following from:
#   compass/ocean/tests/global_ocean/tasks.py

# number of cells at 1000m resolution (with no gutter present). This is used
# as heuristic to scale the number of cells with resolution, in order to
# constrain the resoucres (i.e. `ntasks`) based on the mesh size.
ncells_at_1km_res = 61382


def get_ntasks_from_cell_count(config, cell_count):
    """
    Computes `ntasks` and `min_tasks` based on mesh size (i.e. cell_count)

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case the step belongs to

    cell_count : int
        Number of horizontal cells in the mesh. The value of this parameter is
        caclulated (or approximated) using the functions below.

    Returns
    -------
    ntasks : int
        The target number of MPI tasks

    min_tasks : int
        The minimum number of tasks
    """

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


def exact_cell_count(mesh_filename):
    """
    Get the number of cells from an already generated mesh file.

    Parameters
    ----------
    mesh_filename : str
        A file containing the MPAS mesh (specifically `nCells`)

    Returns
    -------
    cell_count : int
        the number of cells in the mesh
    """
    # get cell count from mesh
    with xr.open_dataset(mesh_filename) as ds:
        cell_count = ds.sizes['nCells']

    return cell_count


def approx_cell_count(resolution, gutter_length):
    """
    Approximate the number of cells based on the resolution ratio squared
    times the number of cells at 1km resolution. Also do a crude area scaling
    to account for the additional cells from the gutter (if present).

    Parameters
    ----------
    resolution : float
        The nominal resolution requested in the configuration file.

    gutter_length: float
        Desired gutter length [m] on the eastern domain.

    Returns
    -------
    cell_count : int
        the number of cells in the mesh
    """
    # Fixed domain lenghts [m] (without gutter)
    Lx = 640e3
    Ly = 80e3
    # Reference domain area [m^2]
    ref_area = Lx * Ly

    # ensure that the requested `gutter_length` is valid.
    # Otherwise set the value to zero
    if (gutter_length < 2. * resolution) and (gutter_length != 0.):
        gutter_length = 0.

    # approximate the number of cells for the desired resolution
    cell_count = int(ncells_at_1km_res * (1000 / resolution)**2)

    # Account for extra cells when gutter is requested
    if gutter_length != 0.:
        # calculate the area of the domain with the gutter
        new_area = (Lx + gutter_length) * Ly
        # scale by the approx cell count by the relative area increase
        # due to presence of the gutter
        cell_count *= new_area / ref_area

    # return the approximate cell count as deterimned at `setup` stage.
    return cell_count
