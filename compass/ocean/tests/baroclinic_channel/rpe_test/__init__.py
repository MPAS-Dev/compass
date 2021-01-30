from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.baroclinic_channel import initial_state, forward
from compass.ocean.tests.baroclinic_channel.rpe_test import analysis
from compass.ocean.tests import baroclinic_channel
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    resolution = testcase['resolution']
    testcase['description'] = 'baroclinic channel {} reference potential '\
                              'energy (RPE)'.format(resolution)

    nus = [1, 5, 10, 20, 200]

    res_params = {'1km': {'cores': 144, 'min_cores': 36,
                          'max_memory': 64000, 'max_disk': 64000},
                  '4km': {'cores': 36, 'min_cores': 8,
                          'max_memory': 16000, 'max_disk': 16000},
                  '10km': {'cores': 8, 'min_cores': 4,
                           'max_memory': 2000, 'max_disk': 2000}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))

    defaults = res_params[resolution]

    subdir = '{}/{}'.format(resolution, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, initial_state, resolution=resolution)

    for index, nu in enumerate(nus):
        name = 'rpe_test_{}_nu_{}'.format(index+1, nu)
        # we pass the defaults for the resolution on as keyword arguments
        step = add_step(testcase, forward, name=name, subdir=name, threads=1,
                        nu=float(nu), resolution=resolution, **defaults)

        # add the local namelist and streams file
        add_namelist_file(
            step, 'compass.ocean.tests.baroclinic_channel.rpe_test',
            'namelist.forward')
        add_streams_file(
            step, 'compass.ocean.tests.baroclinic_channel.rpe_test',
            'streams.forward')

    add_step(testcase, analysis, resolution=resolution, nus=nus)


def configure(testcase, config):
    """
    Modify the configuration options for this test case.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration
    """
    baroclinic_channel.configure(testcase, config)


def run(testcase, test_suite, config, logger):
    """
    Run each step of the test case

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case from the ``collect()``
        function

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the test case
    """
    # just run all the steps in the order they were added
    run_steps(testcase, test_suite, config, logger)
