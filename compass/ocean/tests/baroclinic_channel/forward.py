from compass.io import add_input_file, add_output_file
from compass.namelist import add_namelist_file, add_namelist_options, \
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
    defaults = dict(max_memory=1000, max_disk=1000, threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    step.setdefault('min_cores', step['cores'])

    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.forward')
    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.{}.forward'.format(step['resolution']))
    if 'nu' in step:
        # update the viscosity to the requested value
        options = {'config_mom_del2': '{}'.format(step['nu'])}
        add_namelist_options(step, options)

    add_streams_file(step, 'compass.ocean.tests.baroclinic_channel',
                     'streams.forward')


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
    # generate the namelist and streams files file from the various files and
    # replacements we have collected
    generate_namelist(step, config)
    generate_streams(step, config)

    add_model_as_input(step, config)

    add_input_file(step, filename='init.nc',
                   target='../initial_state/ocean.nc')
    add_input_file(step, filename='graph.info',
                   target='../initial_state/culled_graph.info')

    add_output_file(step, filename='output.nc')


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
    run_model(step, config, logger)
