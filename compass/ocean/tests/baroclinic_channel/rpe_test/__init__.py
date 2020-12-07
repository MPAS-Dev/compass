from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.baroclinic_channel import initial_state, forward
from compass.ocean.tests.baroclinic_channel.rpe_test import analysis
from compass.ocean.tests import baroclinic_channel


def collect(resolution):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    resolution : {'1km', '4km', '10km'}
        The resolution of the mesh

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'baroclinic channel {} reference potential energy (RPE)' \
                  ''.format(resolution)
    module = __name__

    res_params = {'1km': {'core_count': 144},
                  '4km': {'core_count': 36},
                  '10km': {'core_count': 8}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))

    res_params = res_params[resolution]

    procs = res_params['core_count']

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(resolution, name)
    steps = dict()
    step = initial_state.collect(resolution)
    steps[step['name']] = step

    for index, nu in enumerate([1, 5, 10, 20, 200]):
        step = forward.collect(resolution, procs=procs, threads=1,
                               testcase_module=module,
                               namelist_file='namelist.forward',
                               streams_file='streams.forward',
                               nu=float(nu))
        step['name'] = 'rpe_test_{}_nu_{}'.format(index+1, nu)
        step['subdir'] = step['name']
        steps[step['name']] = step

    step = analysis.collect(resolution)
    steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['resolution'] = resolution

    return testcase


def configure(testcase, config):
    """
    Modify the configuration options for this testcase.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    baroclinic_channel.configure(testcase, config)


def run(testcase, test_suite, config, logger):
    """
    Run each step of the testcase

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the testcase
    """
    # just run all the steps in the order they were added
    steps = list(testcase['steps'])
    run_steps(testcase, test_suite, config, steps, logger)
