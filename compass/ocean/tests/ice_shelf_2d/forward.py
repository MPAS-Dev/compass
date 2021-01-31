from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options,\
    generate_namelist
from compass.streams import add_streams_file, generate_streams
from compass.model import add_model_as_input, run_model


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
    with_frazil = step['with_frazil']

    defaults = dict(max_memory=1000, max_disk=1000, threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    add_namelist_file(step, 'compass.ocean.tests.ice_shelf_2d',
                      'namelist.forward')
    if with_frazil:
        options = {'config_use_frazil_ice_formation': '.true.',
                   'config_frazil_maximum_depth': '2000.0'}
        add_namelist_options(step, options)
        add_streams_file(step, 'compass.ocean.streams', 'streams.frazil')

    add_streams_file(step, 'compass.ocean.streams', 'streams.land_ice_fluxes')

    add_streams_file(step, 'compass.ocean.tests.ice_shelf_2d',
                     'streams.forward')

    add_input_file(step, filename='init.nc',
                   target='../ssh_adjustment/adjusted_init.nc')
    add_input_file(step, filename='graph.info',
                   target='../initial_state/culled_graph.info')

    add_output_file(step, 'output.nc')


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
    run_model(step, config, logger)
