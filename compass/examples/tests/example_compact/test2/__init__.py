from compass.config import add_config
from compass.testcase import add_step, set_testcase_subdir, run_steps
from compass.examples.tests.example_compact import step1, step2


# See test1 for more details
def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    # you can get any information out of the "testcase" dictionary, e.g. to
    # pass them on to steps.  Some of the entries will be from the framework
    # while others are passed in as keyword arguments to "add_testcase" in the
    # configuration's "collect()"
    resolution = testcase['resolution']

    # you must add a description
    testcase['description'] = 'Template {} test2'.format(resolution)

    # You can change the subdirectory from the default, the name of the test
    # case.  In this case, we add a directory for the resolution.
    subdir = '{}/{}'.format(resolution, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, step1, resolution=resolution)
    add_step(testcase, step2, resolution=resolution)


# this function can be used to add the contents of a config file as in the
# example below or to add or override specific config options, as also shown
# here.  The function must take only the "testcase" and "config" arguments, so
# any information you need should be added to "testcase" if it is not available
# in one of the config files used to build "config"
def configure(testcase, config):
    """
    Modify the configuration options for this test case.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    # add (or override) some configuration options that will be used during any
    # or all of the steps in this test case
    add_config(config, 'compass.examples.tests.example_compact.test2',
               'test2.cfg')

    # add a config option to the config file
    config.set('example_compact', 'resolution', testcase['resolution'])


# The function must take only the "testcase" and "config" arguments, so
# any information you need in order to run the test case should be added to
# "testcase" if it is not available in "config"
def run(testcase, test_suite, config, logger):
    """
    Run each step of the test case

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the test case
    """
    # typically, this involves running all the steps in the test case in the
    # desired sequence.  However, it may involve only running a subset of steps
    # if some are optional and not performed by default.
    run_steps(testcase, test_suite, config, logger)
