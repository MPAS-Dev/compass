from importlib.resources import path

from compass.io import symlink
from compass.testcase import add_step, run_steps
from compass.config import add_config
from compass.landice.tests.enthalpy_benchmark import setup_mesh, run_model
from compass.landice.tests.enthalpy_benchmark.B import visualize


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    testcase['description'] = 'Kleiner enthalpy benchmark B'

    add_step(testcase, setup_mesh)

    add_step(testcase, run_model, cores=1, threads=1)

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
    add_config(config, 'compass.landice.tests.enthalpy_benchmark.B',
               'B.cfg', exception=True)

    with path('compass.landice.tests.enthalpy_benchmark', 'README') as \
            target:
        symlink(str(target), '{}/README'.format(testcase['work_dir']))


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
