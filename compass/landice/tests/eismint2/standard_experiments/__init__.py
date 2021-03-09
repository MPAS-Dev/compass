from compass.testcase import add_step, run_steps
from compass.landice.tests.eismint2 import setup_mesh, run_experiment
from compass.landice.tests.eismint2.standard_experiments import visualize


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    testcase['description'] = 'EISMINT2 standard experiments'

    add_step(testcase, setup_mesh)

    for experiment in ['a', 'b', 'c', 'd', 'f', 'g']:
        name = 'experiment_{}'.format(experiment)
        add_step(testcase, run_experiment, name=name, subdir=name, cores=4,
                 threads=1, experiment=experiment)

    add_step(testcase, visualize)


def configure(testcase, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    # We want to visualize all test cases by default
    config.set('eismint2_viz', 'experiment', 'a, b, c, d, f, g')


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
