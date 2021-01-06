# make sure to add all meshes here so they will be found in sys.modules below
from compass.ocean.tests.global_ocean.mesh import qu240, ec30to60

from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean.mesh import mesh
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables


def collect(mesh_name, with_ice_shelf_cavities):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh should include ice-shelf cavities

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'global ocean {} - mesh creation'.format(mesh_name)
    module = __name__

    name = module.split('.')[-1]
    subdir = '{}/{}'.format(mesh_name, name)
    steps = dict()
    step = mesh.collect(mesh_name, cores=4, min_cores=2,
                        max_memory=1000, max_disk=1000, threads=1,
                        with_ice_shelf_cavities=with_ice_shelf_cavities)
    steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name

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

    variables = ['xCell', 'yCell', 'zCell']
    compare_variables(variables, config, testcase['work_dir'],
                      filename1='mesh/culled_mesh.nc')
