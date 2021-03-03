# make sure to add all meshes here so they will be found in sys.modules below
from compass.ocean.tests.global_ocean.mesh import qu240, ec30to60, so12to60

from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean.mesh import mesh
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables


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
    testcase['description'] = \
        'global ocean {} - mesh creation'.format(mesh_name)

    subdir = '{}/{}'.format(mesh_name, testcase['name'])
    set_testcase_subdir(testcase, subdir)

    add_step(testcase, mesh, mesh_name=mesh_name,
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
    step = testcase['steps']['mesh']
    # get the these properties from the config options
    for option in ['cores', 'min_cores', 'max_memory', 'max_disk']:
        step[option] = config.getint('global_ocean',
                                     'mesh_{}'.format(option))

    run_steps(testcase, test_suite, config, logger)

    variables = ['xCell', 'yCell', 'zCell']
    compare_variables(variables, config, testcase['work_dir'],
                      filename1='mesh/culled_mesh.nc')
