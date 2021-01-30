from importlib.resources import path

from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm import \
    ocean_initial_condition, ocean_graph_partition, seaice_initial_condition, \
    scrip


def collect(testcase):
    """
    Update the dictionary of test case properties and add steps

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which can be updated
    """
    mesh_name = testcase['mesh_name']
    with_ice_shelf_cavities = testcase['with_ice_shelf_cavities']
    initial_condition = testcase['initial_condition']
    with_bgc = testcase['with_bgc']
    time_integrator = testcase['time_integrator']
    name = testcase['name']
    restart_filename = testcase['restart_filename']

    testcase['description'] = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='files for E3SM')

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)
    set_testcase_subdir(testcase, subdir)

    restart_filename = '../../spinup/{}/{}'.format(time_integrator,
                                                   restart_filename)

    add_step(testcase, ocean_initial_condition, mesh_name=mesh_name,
             restart_filename=restart_filename)

    add_step(testcase, ocean_graph_partition, mesh_name=mesh_name,
             restart_filename=restart_filename)

    add_step(testcase, seaice_initial_condition, mesh_name=mesh_name,
             restart_filename=restart_filename,
             with_ice_shelf_cavities=with_ice_shelf_cavities)

    add_step(testcase, scrip, mesh_name=mesh_name,
             restart_filename=restart_filename,
             with_ice_shelf_cavities=with_ice_shelf_cavities)


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
        A dictionary of properties of this test case

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the test case
    """
    run_steps(testcase, test_suite, config, logger)
