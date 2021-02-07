from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.subdir import get_forward_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.namelist import add_namelist_options
from compass.streams import add_streams_file
from compass.io import add_input_file, add_output_file


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
    module = __name__

    if time_integrator != 'split_explicit':
        raise ValueError('{} spin-up not defined for {}'.format(
            mesh_name, time_integrator))

    testcase['description'] = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='spin-up')

    subdir = get_forward_sudbdir(mesh_name, initial_condition, with_bgc,
                                 time_integrator, name)
    set_testcase_subdir(testcase, subdir)

    restart_times = ['0001-01-11_00:00:00', '0001-01-21_00:00:00']
    restart_filenames = [
        'restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
        for restart_time in restart_times]

    # first spin-up step
    step_name = 'damped_spinup_1'
    step = add_step(testcase, forward, name=step_name, subdir=step_name,
                    mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    namelist_options = {
        'config_run_duration': "'00-00-10_00:00:00'",
        'config_dt': "'00:20:00'",
        'config_Rayleigh_friction': '.true.',
        'config_Rayleigh_damping_coeff': '1.0e-4'}
    add_namelist_options(step, namelist_options)

    stream_replacements = {
        'output_interval': '00-00-10_00:00:00',
        'restart_interval': '00-00-10_00:00:00'}
    add_streams_file(step, module, 'streams.template',
                     template_replacements=stream_replacements)

    add_output_file(step, filename='../{}'.format(restart_filenames[0]))

    # final spin-up step
    step_name = 'simulation'
    step = add_step(testcase, forward, name=step_name, subdir=step_name,
                    mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    namelist_options = {
        'config_run_duration': "'00-00-10_00:00:00'",
        'config_do_restart': '.true.',
        'config_start_time': "'{}'".format(restart_times[0])}
    add_namelist_options(step,  namelist_options)

    stream_replacements = {
        'output_interval': '00-00-10_00:00:00',
        'restart_interval': '00-00-10_00:00:00'}
    add_streams_file(step, module, 'streams.template',
                     template_replacements=stream_replacements)

    add_input_file(step, filename='../{}'.format(restart_filenames[0]))
    add_output_file(step, filename='../{}'.format(restart_filenames[1]))

    testcase['restart_filenames'] = restart_filenames


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
