import os
import xarray
import numpy as np
from glob import glob

from mpas_tools.logging import check_call

from compass.testcase import get_step_default
from compass.io import symlink
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path


def collect(mesh_name, restart_filename, cores=1, min_cores=None,
            max_memory=1000, max_disk=1000, threads=1):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    restart_filename : str
        The relative path to a restart file to use as the initial condition
        for E3SM

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
    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))

    inputs = []
    links = {'{}/culled_graph.info'.format(mesh_path): 'graph.info',
             '../{}'.format(step['restart_filename']): 'restart.nc'}

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
    with xarray.open_dataset('restart.nc') as ds:
        mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
        mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
        prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
        creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

    try:
        os.makedirs('../assembled_files/inputdata/ocn/mpas-o/{}'.format(
            mesh_short_name))
    except OSError:
        pass

    symlink('graph.info', 'mpas-o.graph.info.{}'.format(creation_date))

    nCells = sum(1 for _ in open('graph.info'))
    min_graph_size = int(nCells / 6000)
    max_graph_size = int(nCells / 100)
    logger.info('Creating graph files between {} and {}'.format(
        min_graph_size, max_graph_size))
    n_power2 = 2**np.arange(1, 21)
    n_multiples12 = 12 * np.arange(1, 9)

    n = n_power2
    for power10 in range(3):
        n = np.concatenate([n, 10**power10 * n_multiples12])

    for index in range(len(n)):
        if min_graph_size <= n[index] <= max_graph_size:
            args = ['gpmetis', 'mpas-o.graph.info.{}'.format(creation_date),
                    '{}'.format(n[index])]
            check_call(args, logger)

    # create link in assembled files directory
    files = glob('mpas-o.graph.info.*')
    dest_path = '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
        mesh_short_name)
    for file in files:
        symlink('../../../../../ocean_graph_partition/{}'.format(file),
                '{}/{}'.format(dest_path, file))
