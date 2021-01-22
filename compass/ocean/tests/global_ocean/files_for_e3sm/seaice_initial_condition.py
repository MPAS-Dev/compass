import os
import xarray

from mpas_tools.io import write_netcdf

from compass.testcase import get_step_default
from compass.io import symlink


def collect(mesh_name, restart_filename, with_ice_shelf_cavities, cores=1,
            min_cores=None, max_memory=1000, max_disk=1000, threads=1):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    restart_filename : str
        The relative path to a restart file to use as the initial condition
        for E3SM

    with_ice_shelf_cavities : bool
        Whether the mesh should include ice-shelf cavities

    cores : int, optional
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
    step['restart_filename'] = restart_filename
    step['with_ice_shelf_cavities'] = with_ice_shelf_cavities

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

    symlink('../README', '{}/README'.format(step_dir))

    inputs = []
    links = {'../{}'.format(step['restart_filename']): 'restart.nc'}

    for target, link in links.items():
        symlink(target, os.path.join(step_dir, link))
        inputs.append(os.path.abspath(os.path.join(step_dir, target)))

    step['inputs'] = inputs

    # for now, we won't define any outputs because they include the mesh short
    # name, which is not known at setup time.  Currently, this is safe because
    # no other steps depend on the outputs of this one.
    step['outputs'] = []


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
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']

    with xarray.open_dataset('restart.nc') as ds:
        mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']

    try:
        os.makedirs('../assembled_files/inputdata/ocn/mpas-cice/{}'.format(
            mesh_short_name))
    except OSError:
        pass

    restart_filename = os.path.abspath(
        os.path.join('..', step['restart_filename']))
    source_filename = '{}.nc'.format(mesh_short_name)
    dest_filename = 'seaice.{}.nc'.format(mesh_short_name)

    keep_vars = ['areaCell', 'cellsOnCell', 'edgesOnCell', 'fCell',
                 'indexToCellID', 'latCell', 'lonCell', 'meshDensity',
                 'nEdgesOnCell', 'verticesOnCell', 'xCell', 'yCell', 'zCell',
                 'angleEdge', 'cellsOnEdge', 'dcEdge', 'dvEdge', 'edgesOnEdge',
                 'fEdge', 'indexToEdgeID', 'latEdge', 'lonEdge',
                 'nEdgesOnCell', 'nEdgesOnEdge', 'verticesOnEdge',
                 'weightsOnEdge', 'xEdge', 'yEdge', 'zEdge', 'areaTriangle',
                 'cellsOnVertex', 'edgesOnVertex', 'fVertex',
                 'indexToVertexID', 'kiteAreasOnVertex', 'latVertex',
                 'lonVertex', 'xVertex', 'yVertex', 'zVertex']

    if with_ice_shelf_cavities:
        keep_vars.append('landIceMask')

    symlink(restart_filename, source_filename)
    with xarray.open_dataset(source_filename) as ds:
        ds.load()
        ds = ds[keep_vars]
        write_netcdf(ds, dest_filename)

    symlink('../../../../../seaice_initial_condition/{}'.format(dest_filename),
            '../assembled_files/inputdata/ocn/mpas-cice/{}/{}'.format(
                mesh_short_name, dest_filename))
