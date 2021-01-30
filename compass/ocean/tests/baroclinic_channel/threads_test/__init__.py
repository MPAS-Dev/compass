from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.baroclinic_channel import initial_state, forward
from compass.ocean.tests import baroclinic_channel
from compass.validate import compare_variables


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    resolution = testcase['resolution']
    testcase['description'] = \
        'baroclinic channel {} thread test'.format(resolution)

    subdir = '{}/{}'.format(resolution, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, initial_state, resolution=resolution)

    for threads in [1, 2]:
        name = '{}thread'.format(threads)
        add_step(testcase, forward, name=name, subdir=name, cores=4,
                 threads=threads, resolution=resolution)


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
    run_steps(testcase, test_suite, config, logger)
    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    steps = testcase['steps_to_run']
    if '1thread' in steps and '2thread' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='1thread/output.nc',
                          filename2='2thread/output.nc')
