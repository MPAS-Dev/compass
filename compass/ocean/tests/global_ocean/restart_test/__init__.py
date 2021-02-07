from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.subdir import get_forward_sudbdir
from compass.namelist import add_namelist_file
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

    testcase['description'] = get_description(
        mesh_name, initial_condition, with_bgc, time_integrator,
        description='restart test')

    subdir = get_forward_sudbdir(mesh_name, initial_condition, with_bgc,
                                 time_integrator, name)
    set_testcase_subdir(testcase, subdir)

    restart_time = {'split_explicit': '0001-01-01_04:00:00',
                    'RK4': '0001-01-01_00:10:00'}
    restart_filename = '../restarts/rst.{}.nc'.format(
        restart_time[time_integrator].replace(':', '.'))
    input_file = {'restart': restart_filename}
    output_file = {'full': restart_filename}
    for part in ['full', 'restart']:
        name = '{}_run'.format(part)
        step = add_step(testcase, forward, name=name, subdir=name, cores=4,
                        threads=1, mesh_name=mesh_name,
                        with_ice_shelf_cavities=with_ice_shelf_cavities,
                        initial_condition=initial_condition, with_bgc=with_bgc,
                        time_integrator=time_integrator)

        suffix = '{}.{}'.format(time_integrator.lower(), part)
        add_namelist_file(step, module, 'namelist.{}'.format(suffix))
        add_streams_file(step, module, 'streams.{}'.format(suffix))
        if part in input_file:
            add_input_file(step, filename=input_file[part])
        if part in output_file:
            add_output_file(step, filename=output_file[part])


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
    if 'full_run' in steps and 'restart_run' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')
