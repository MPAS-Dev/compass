import sys

from mpas_tools.ocean import build_spherical_mesh

from compass.ocean.tests.global_ocean.mesh.cull import cull_mesh
from compass.io import add_output_file


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(cores=1, min_cores=1, max_memory=8000, max_disk=8000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core, configuration and test case
    """
    for file in ['culled_mesh.nc', 'culled_graph.info',
                 'critical_passages_mask_final.nc']:
        add_output_file(step, filename=file)


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    mesh_name = step['mesh_name']
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']
    # only use progress bars if we're not writing to a log file
    use_progress_bar = 'log_filename' not in step

    # create the base mesh
    cellWidth, lon, lat = build_cell_width_lat_lon(mesh_name)
    build_spherical_mesh(cellWidth, lon, lat, out_filename='base_mesh.nc',
                         logger=logger, use_progress_bar=use_progress_bar)

    cull_mesh(with_critical_passages=True, logger=logger,
              use_progress_bar=use_progress_bar,
              with_cavities=with_ice_shelf_cavities)


def build_cell_width_lat_lon(mesh_name):
    """
    Create cell width array for this mesh on a regular latitude-longitude grid

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    Returns
    -------
    cellWidth : numpy.array
        m x n array of cell width in km

    lon : numpy.array
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.array
        longitude in degrees (length m and between -90 and 90)
    """

    package, _ = get_mesh_package(mesh_name)
    build_cell_width = getattr(sys.modules[package],
                               'build_cell_width_lat_lon')
    return build_cell_width()


def get_mesh_package(mesh_name):
    """
    Get the system module corresponding to the given mesh name

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    Returns
    -------
    module : Package
        The system module for the given mesh, one of the packages in
        ``compass.ocean.tests.global_ocean.mesh`` with the mesh name converted
        to lowercase

    prefix : str
        The prefix of the package (the mesh name as lowercase and with 'wisc'
        suffix removed)

    Raises
    ------
    ValueError
        If the corresponding module for the given mesh does not exist

    """
    prefix = mesh_name.lower().replace('wisc', '')
    package = 'compass.ocean.tests.global_ocean.mesh.{}'.format(prefix)
    if package not in sys.modules:
        raise ValueError('Mesh {} missing corresponding package {}'.format(
            mesh_name, package))

    return package, prefix
