from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir


def collect(mesh_name, with_ice_shelf_cavities, initial_condition, with_bgc,
            time_integrator):
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

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='restart test')
    module = __name__

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)

    restart_time = {'split_explicit': '0001-01-01_04:00:00',
                    'RK4': '0001-01-01_00:10:00'}
    restart_filename = '../restarts/rst.{}.nc'.format(
        restart_time[time_integrator])
    inputs = {'full': None, 'restart': [restart_filename]}
    outputs = {'full': ['output.nc', restart_filename], 'restart': None}
    steps = dict()
    for step_prefix in ['full', 'restart']:
        suffix = '{}.{}'.format(time_integrator.lower(), step_prefix)
        step = forward.collect(mesh_name, with_ice_shelf_cavities, with_bgc,
                               time_integrator, cores=4, threads=1,
                               testcase_module=module,
                               namelist_file='namelist.{}'.format(suffix),
                               streams_file='streams.{}'.format(suffix),
                               inputs=inputs[step_prefix],
                               outputs=outputs[step_prefix])
        step['name'] = '{}_run'.format(step_prefix)
        step['subdir'] = step['name']
        steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name
    testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities
    testcase['initial_condition'] = initial_condition
    testcase['with_bgc'] = with_bgc

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
    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    steps = testcase['steps_to_run']
    if 'full_run' in steps and 'restart_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
