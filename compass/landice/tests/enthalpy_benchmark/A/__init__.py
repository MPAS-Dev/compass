from importlib.resources import path

from compass.io import symlink
from compass.testcase import add_step, run_steps
from compass.config import add_config
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file
from compass.validate import compare_variables
from compass.landice.tests.enthalpy_benchmark import setup_mesh, run_model
from compass.landice.tests.enthalpy_benchmark.A import visualize


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    testcase['description'] = 'Kleiner enthalpy benchmark A'

    module = __name__

    add_step(testcase, setup_mesh)

    restart_filenames = ['../setup_mesh/landice_grid.nc',
                         '../phase1/restart.100000.nc',
                         '../phase2/restart.150000.nc']
    for index, restart_filename in enumerate(restart_filenames):
        name = 'phase{}'.format(index+1)
        step = add_step(testcase, run_model, cores=1, threads=1, name=name,
                        subdir=name, restart_filename=restart_filename)

        suffix = 'landice{}'.format(index+1)
        add_namelist_file(step, module, 'namelist.{}'.format(suffix))
        add_streams_file(step, module, 'streams.{}'.format(suffix))

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
    add_config(config, 'compass.landice.tests.enthalpy_benchmark.A',
               'A.cfg', exception=True)

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
    variables = ['temperature', 'basalWaterThickness', 'groundedBasalMassBal']
    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='phase3/output.nc')
