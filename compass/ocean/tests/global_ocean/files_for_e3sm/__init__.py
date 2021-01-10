from importlib.resources import path

from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm import \
    ocean_initial_condition, ocean_graph_partition, seaice_initial_condition, \
    scrip


def collect(mesh_name, with_ice_shelf_cavities, initial_condition, with_bgc,
            time_integrator, restart_filename):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh should include ice-shelf cavities

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition to build

    with_bgc : bool
        Whether to include BGC variables in the initial condition

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the run

    restart_filename : str
        The relative path to a restart file to use as the initial condition
        for E3SM

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='files for E3SM')
    module = __name__

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)

    restart_filename = '../../spinup/{}/{}'.format(time_integrator,
                                                   restart_filename)

    steps = dict()
    step = ocean_initial_condition.collect(mesh_name, restart_filename)
    steps[step['name']] = step
    step = ocean_graph_partition.collect(mesh_name, restart_filename)
    steps[step['name']] = step
    step = seaice_initial_condition.collect(mesh_name, restart_filename)
    steps[step['name']] = step
    step = scrip.collect(mesh_name, restart_filename, with_ice_shelf_cavities)
    steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name
    testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities

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
    global_ocean.configure(testcase, config)
    with path('compass.ocean.tests.global_ocean.files_for_e3sm', 'README') as \
            target:
        symlink(str(target), '{}/README'.format(testcase['work_dir']))


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
    run_steps(testcase, test_suite, config, logger)
