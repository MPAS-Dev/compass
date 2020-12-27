import os

from mpas_tools.ocean import build_spherical_mesh

from compass.testcase import get_step_default
from compass.ocean.tests.global_ocean.cull import cull_mesh
from compass.ocean.tests.global_ocean.mesh import build_cell_width_lat_lon


def collect(mesh_name, cores, min_cores=None, max_memory=1000,
            max_disk=1000, threads=1):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    cores : int
        The number of cores to run on in init runs. If this many cores are
        available on the machine or batch job, the task will run on that
        number. If fewer are available (but no fewer than min_cores), the job
        will run on all available cores instead.

    min_cores : int, optional
        The minimum allowed cores.  If that number of cores are not available
        on the machine or in the batch job, the run will fail.  By default,
        ``min_cores = cores``

    max_memory : int, optional
        The maximum amount of memory (in MB) this step is allowed to use

    max_disk : int, optional
        The maximum amount of disk space  (in MB) this step is allowed to use

    threads : int, optional
        The number of threads to run with during init runs

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_step_default(__name__)
    step['mesh_name'] = mesh_name
    step['cores'] = cores
    step['max_memory'] = max_memory
    step['max_disk'] = max_disk
    if min_cores is None:
        min_cores = cores
    step['min_cores'] = min_cores
    step['threads'] = threads

    return step


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core, configuration and testcase
    """
    step_dir = step['work_dir']

    inputs = []
    outputs = []

    for file in ['culled_mesh.nc', 'culled_graph.info',
                 'critical_passages_mask_final.nc']:
        outputs.append(os.path.join(step_dir, file))

    step['inputs'] = inputs
    step['outputs'] = outputs


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
    """
    mesh_name = step['mesh_name']

    # create the base mesh
    cellWidth, lon, lat = build_cell_width_lat_lon(mesh_name)
    build_spherical_mesh(cellWidth, lon, lat, out_filename='base_mesh.nc')

    cull_mesh(with_critical_passages=True)
