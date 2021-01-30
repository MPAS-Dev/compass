from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir


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

    testcase['description'] = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='decomposition test')

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)
    set_testcase_subdir(testcase, subdir)

    for procs in [4, 8]:
        name = '{}proc'.format(procs)
        add_step(testcase, forward, name=name, subdir=name, cores=procs,
                 threads=1, mesh_name=mesh_name,
                 with_ice_shelf_cavities=with_ice_shelf_cavities,
                 initial_condition=initial_condition, with_bgc=with_bgc,
                 time_integrator=time_integrator)


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
    run_steps(testcase, test_suite, config, logger)
    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    steps = testcase['steps_to_run']
    if '4proc' in steps and '8proc' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='4proc/output.nc',
                          filename2='8proc/output.nc')
