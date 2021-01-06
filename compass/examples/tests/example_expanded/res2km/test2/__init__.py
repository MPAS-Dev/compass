from compass.config import add_config
from compass.testcase import get_testcase_default, run_steps
from compass.examples.tests.example_expanded.res2km.test2 import step1, step2


def collect():
    """
    Get a dictionary of testcase properties

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    # fill in a useful description of the test case
    description = 'Tempate 2km test2'
    module = __name__
    resolution = '2km'

    # the name of the testcase is the last part of the Python module (the
    # folder it's in, so "test1" or "test2" in the "example_expanded"
    # configuration
    name = module.split('.')[-1]
    # A subdirectory for the testcase after setup.  This can be anything that
    # will ensure that the testcase ends up in a unique directory
    subdir = '{}/{}'.format(resolution, name)
    # make a dictionary of steps for this testcase by calling each step's
    # "collect" function
    steps = dict()
    for step_module in [step1, step2]:
        step = step_module.collect()
        steps[step['name']] = step

    # get some default information for the testcase
    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    # add any parameters or other information you would like to have when you
    # are setting up or running the testcase or its steps
    testcase['resolution'] = resolution

    return testcase


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
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    # add (or override) some configuration options that will be used during any
    # or all of the steps in this testcase
    add_config(config, 'compass.examples.tests.example_expanded.res2km.test2',
               'test2.cfg')

    # add a config option to the config file
    config.set('example_expanded', 'resolution', testcase['resolution'])


# The function must take only the "testcase" and "config" arguments, so
# any information you need in order to run the testcase should be added to
# "testcase" if it is not available in "config"
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
    # typically, this involves running all the steps in the testcase in the
    # desired sequence.  However, it may involve only running a subset of steps
    # if some are optional and not performed by default.
    run_steps(testcase, test_suite, config, logger)
