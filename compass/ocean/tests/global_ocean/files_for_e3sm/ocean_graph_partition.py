import os
import xarray
import numpy as np
from glob import glob

from mpas_tools.logging import check_call

from compass.io import symlink, add_input_file
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path


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
    defaults = dict(cores=1, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))

    add_input_file(step, filename='README', target='../README')
    add_input_file(step, filename='graph.info',
                   target='{}/culled_graph.info'.format(mesh_path))
    add_input_file(step, filename='restart.nc',
                   target='../{}'.format(step['restart_filename']))

    # for now, we won't define any outputs because they include the mesh short
    # name, which is not known at setup time.  Currently, this is safe because
    # no other steps depend on the outputs of this one.


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

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
