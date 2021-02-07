from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.subdir import get_forward_sudbdir
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file


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
        description='daily averaged output test')

    subdir = get_forward_sudbdir(mesh_name, initial_condition, with_bgc,
                                 time_integrator, name)
    set_testcase_subdir(testcase, subdir)

    step = add_step(testcase, forward, cores=4, threads=1, mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    module = __name__
    add_namelist_file(step, module, 'namelist.forward')
    add_streams_file(step, module, 'streams.forward')


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
    work_dir = testcase['work_dir']
    run_steps(testcase, test_suite, config, logger)

    variables = [
        'timeDaily_avg_activeTracers_temperature',
        'timeDaily_avg_activeTracers_salinity',
        'timeDaily_avg_layerThickness', 'timeDaily_avg_normalVelocity',
        'timeDaily_avg_ssh']

    compare_variables(
        variables, config, work_dir=work_dir,
        filename1='forward/analysis_members/'
                  'mpaso.hist.am.timeSeriesStatsDaily.0001-01-01.nc')
