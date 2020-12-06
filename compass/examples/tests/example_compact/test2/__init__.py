from compass.examples.tests.example_compact.testcase import collect as \
    collect_testcase

from compass.config import add_config
from compass.testcase import run_steps


# "resolution" is just an example argument.  The argument can be any parameter
# that distinguishes different variants of a test
def collect(resolution):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    resolution : {'1km', '2km'}
        The resolution of the mesh

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    # fill in a useful description of the test case
    description = 'Tempate {} test2'.format(resolution)
    # This example assumes that it is possible to call a "collect" function
    # that is generic to all testcases with a different parameter ("resolution"
    # in this case).
    testcase = collect_testcase(__name__, description, resolution)
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
    add_config(config, 'compass.examples.tests.example_compact.test2',
               'test2.cfg')

    # add a config option to the config file
    config.set('example_compact', 'resolution', testcase['resolution'])


# The function must take only the "testcase" and "config" arguments, so
# any information you need in order to run the testcase should be added to
# "testcase" if it is not available in "config"
def run(testcase, test_suite, config):
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
    """
    # typically, this involves running all the steps in the testcase in the
    # desired sequence.  However, it may involve only running a subset of steps
    # if some are optional and not performed by default.
    steps = ['step1', 'step2']
    run_steps(testcase, test_suite, config, steps)
