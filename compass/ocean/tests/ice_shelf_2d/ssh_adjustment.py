import os

from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options,\
    generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.ocean.iceshelf import adjust_ssh
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
    defaults = dict(max_memory=1000, max_disk=1000)
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    # generate the namelist, replacing a few default options
    # start with the same namelist settings as the forward run
    add_namelist_file(step, 'compass.ocean.tests.ice_shelf_2d',
                      'namelist.forward')

    # we don't want the global stats AM for this run
    add_namelist_options(step, {'config_AM_globalStats_enable': '.false.'})

    # we want a shorter run and no freshwater fluxes under the ice shelf from
    # these namelist options
    add_namelist_file(step, 'compass.ocean.namelists', 'namelist.ssh_adjust')

    add_streams_file(step, 'compass.ocean.streams', 'streams.ssh_adjust')

    add_input_file(step, filename='adjusting_init0.nc',
                   target='../initial_state/initial_state.nc')

    add_input_file(step, filename='graph.info',
                   target='../initial_state/culled_graph.info')

    add_output_file(step, filename='adjusted_init.nc')


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
    generate_namelist(step, config, mode='forward')
    generate_streams(step, config, mode='forward')

    add_model_as_input(step, config)


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
