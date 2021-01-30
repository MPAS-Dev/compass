import os

from compass.io import symlink, add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options, \
    generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.ocean.iceshelf import adjust_ssh
from compass.ocean.tests.global_ocean.subdir import get_mesh_relative_path, \
    get_initial_condition_relative_path
from compass.model import add_model_as_input


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
    defaults = dict(max_memory=8000, max_disk=8000, threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    if 'cores' in step:
        step.setdefault('min_cores', step['cores'])


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    step_dir = step['work_dir']

    # generate namelist file
    add_namelist_file(
        step, 'compass.ocean.tests.global_ocean', 'namelist.forward')
    add_namelist_options(step, {'config_AM_globalStats_enable': '.false.'})
    add_namelist_file(step, 'compass.ocean.namelists', 'namelist.ssh_adjust')
    generate_namelist(step, config, mode='forward')

    # generate the streams file
    add_streams_file(step,  'compass.ocean.streams', 'streams.ssh_adjust')
    generate_streams(step, config, mode='forward')

    add_model_as_input(step, config)

    mesh_path = '{}/mesh/mesh'.format(get_mesh_relative_path(step))
    init_path = '{}/init/initial_state'.format(
        get_initial_condition_relative_path(step))

    add_input_file(step, filename='adjusting_init0.nc',
                   target='{}/initial_state.nc'.format(init_path))
    add_input_file(step, filename='forcing_data.nc',
                   target='{}/init_mode_forcing_data.nc'.format(init_path))
    add_input_file(step, filename='graph.info',
                   target='{}/culled_graph.info'.format(mesh_path))

    add_output_file(step, filename='adjusted_init.nc')

    # get the these properties from the config options
    for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                   'threads']:
        step[option] = config.getint('global_ocean',
                                     'forward_{}'.format(option))


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
    iteration_count = config.getint('ssh_adjustment', 'iterations')
    adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
               step=step, config=config, logger=logger)
