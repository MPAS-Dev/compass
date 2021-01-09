from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables


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
    if time_integrator != 'split_explicit':
        raise ValueError('{} spin-up not defined for {}'.format(
            mesh_name, time_integrator))

    description = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='spin-up')
    module = __name__

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)

    steps = dict()

    restart_times = ['0001-01-11_00:00:00']
    restart_filenames = [
        '../restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
        for restart_time in restart_times]

    step_name = 'damped_spinup_1'
    inputs = None
    outputs = ['output.nc', restart_filenames[0]]
    namelist_replacements = {
        'config_run_duration': "'00-00-10_00:00:00'",
        'config_dt': "'00:20:00'",
        'config_Rayleigh_friction': '.true.',
        'config_Rayleigh_damping_coeff': '1.0e-4'}
    stream_replacements = {
        'output_interval': '00-00-10_00:00:00',
        'restart_interval': '00-00-10_00:00:00'}
    step = forward.collect(mesh_name, with_ice_shelf_cavities,
                           with_bgc,  time_integrator,
                           testcase_module=module,
                           streams_file='streams.template',
                           namelist_replacements=namelist_replacements,
                           stream_replacements=stream_replacements,
                           inputs=inputs, outputs=outputs)
    step['name'] = step_name
    step['subdir'] = step['name']
    steps[step['name']] = step

    step_name = 'simulation'
    inputs = [restart_filenames[0]]
    outputs = None
    namelist_replacements = {
        'config_run_duration': "'00-00-10_00:00:00'",
        'config_do_restart': '.true.',
        'config_start_time': "'{}'".format(restart_times[0])}
    stream_replacements = {
        'output_interval': '00-00-10_00:00:00',
        'restart_interval': '00-00-10_00:00:00'}
    step = forward.collect(mesh_name, with_ice_shelf_cavities,
                           with_bgc,  time_integrator,
                           testcase_module=module,
                           streams_file='streams.template',
                           namelist_replacements=namelist_replacements,
                           stream_replacements=stream_replacements,
                           inputs=inputs, outputs=outputs)
    step['name'] = step_name
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
    # get the these properties from the config options
    for step_name in testcase['steps_to_run']:
        step = testcase['steps'][step_name]
        for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                       'threads']:
            step[option] = config.getint('global_ocean',
                                         'forward_{}'.format(option))

    run_steps(testcase, test_suite, config, logger)

    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']

    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='simulation/output.nc')
