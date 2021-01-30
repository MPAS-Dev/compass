from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.baroclinic_channel import initial_state, forward
from compass.ocean.tests import baroclinic_channel
from compass.validate import compare_variables
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
    testcase['description'] = \
        'baroclinic channel {} restart test'.format(resolution)

    subdir = '{}/{}'.format(resolution, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, initial_state, resolution=resolution)

    for part in ['full', 'restart']:
        name = '{}_run'.format(part)
        step = add_step(testcase, forward, name=name, subdir=name, cores=4,
                        threads=1, resolution=resolution)

        # add the local namelist and streams file
        add_namelist_file(
            step, 'compass.ocean.tests.baroclinic_channel.restart_test',
            'namelist.{}'.format(part))
        add_streams_file(
            step, 'compass.ocean.tests.baroclinic_channel.restart_test',
            'streams.{}'.format(part))


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
    if 'full_run' in steps and 'restart_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
