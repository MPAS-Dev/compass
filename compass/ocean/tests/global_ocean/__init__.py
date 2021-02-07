from compass.ocean.tests.global_ocean import mesh, init, performance_test, \
    restart_test, decomp_test, threads_test, analysis_test, \
    daily_output_test, files_for_e3sm
from compass.ocean.tests.global_ocean.mesh.qu240 import spinup as qu240_spinup
from compass.ocean.tests.global_ocean.mesh.ec30to60 import spinup as \
    ec30to60_spinup
from compass.config import add_config
from compass.ocean.tests.global_ocean.mesh.mesh import get_mesh_package
from compass.ocean.tests.global_ocean.init import add_descriptions_to_config
from compass.testcase import add_testcase


def collect():
    """
    Get a list of test cases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()

    # we do a lot of tests for QU240/QU240wISC
    for mesh_name, with_ice_shelf_cavities in [('QU240', False),
                                               ('QUwISC240', True)]:
        add_testcase(testcases, mesh, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities)

        initial_condition = 'PHC'
        with_bgc = False
        time_integrator = 'split_explicit'
        add_testcase(testcases, init, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities,
                     initial_condition=initial_condition,
                     with_bgc=with_bgc)
        for test in [performance_test, restart_test, decomp_test,
                     threads_test, analysis_test, daily_output_test]:
            add_testcase(
                testcases, test, mesh_name=mesh_name,
                with_ice_shelf_cavities=with_ice_shelf_cavities,
                initial_condition=initial_condition,
                with_bgc=with_bgc, time_integrator=time_integrator)

        testcase = add_testcase(
            testcases, qu240_spinup, mesh_name=mesh_name,
            with_ice_shelf_cavities=with_ice_shelf_cavities,
            initial_condition=initial_condition,
            with_bgc=with_bgc, time_integrator=time_integrator)

        restart_filename = testcase['restart_filenames'][-1]
        add_testcase(
            testcases, files_for_e3sm, mesh_name=mesh_name,
            with_ice_shelf_cavities=with_ice_shelf_cavities,
            initial_condition=initial_condition,
            with_bgc=with_bgc, time_integrator=time_integrator,
            restart_filename=restart_filename)

        time_integrator = 'RK4'
        for test in [performance_test, restart_test, decomp_test,
                     threads_test]:
            add_testcase(
                testcases, test, mesh_name=mesh_name,
                with_ice_shelf_cavities=with_ice_shelf_cavities,
                initial_condition=initial_condition,
                with_bgc=with_bgc, time_integrator=time_integrator)

        time_integrator = 'split_explicit'
        initial_condition = 'EN4_1900'
        with_bgc = False
        add_testcase(testcases, init, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities,
                     initial_condition=initial_condition,
                     with_bgc=with_bgc)
        add_testcase(testcases, performance_test, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities,
                     initial_condition=initial_condition,
                     with_bgc=with_bgc,
                     time_integrator=time_integrator)
        testcase = add_testcase(
            testcases, qu240_spinup, mesh_name=mesh_name,
            with_ice_shelf_cavities=with_ice_shelf_cavities,
            initial_condition=initial_condition,
            with_bgc=with_bgc, time_integrator=time_integrator)

        restart_filename = testcase['restart_filenames'][-1]
        add_testcase(
            testcases, files_for_e3sm, mesh_name=mesh_name,
            with_ice_shelf_cavities=with_ice_shelf_cavities,
            initial_condition=initial_condition,
            with_bgc=with_bgc, time_integrator=time_integrator,
            restart_filename=restart_filename)

        with_bgc = True
        initial_condition = 'PHC'
        add_testcase(testcases, init, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities,
                     initial_condition=initial_condition,
                     with_bgc=with_bgc)
        add_testcase(testcases, performance_test, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities,
                     initial_condition=initial_condition,
                     with_bgc=with_bgc,
                     time_integrator=time_integrator)

    # for other meshes, we do fewer tests
    for mesh_name, with_ice_shelf_cavities in [('EC30to60', False),
                                               ('ECwISC30to60', True)]:
        add_testcase(testcases, mesh, mesh_name=mesh_name,
                     with_ice_shelf_cavities=with_ice_shelf_cavities)

        for initial_condition in ['PHC', 'EN4_1900']:
            with_bgc = False
            time_integrator = 'split_explicit'
            add_testcase(testcases, init, mesh_name=mesh_name,
                         with_ice_shelf_cavities=with_ice_shelf_cavities,
                         initial_condition=initial_condition,
                         with_bgc=with_bgc)
            add_testcase(testcases, performance_test, mesh_name=mesh_name,
                         with_ice_shelf_cavities=with_ice_shelf_cavities,
                         initial_condition=initial_condition,
                         with_bgc=with_bgc, time_integrator=time_integrator)
            testcase = add_testcase(
                testcases, ec30to60_spinup, mesh_name=mesh_name,
                with_ice_shelf_cavities=with_ice_shelf_cavities,
                initial_condition=initial_condition,
                with_bgc=with_bgc, time_integrator=time_integrator)

            restart_filename = testcase['restart_filenames'][-1]
            add_testcase(testcases, files_for_e3sm, mesh_name=mesh_name,
                         with_ice_shelf_cavities=with_ice_shelf_cavities,
                         initial_condition=initial_condition,
                         with_bgc=with_bgc, time_integrator=time_integrator,
                         restart_filename=restart_filename)

    return testcases


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
    mesh_name = testcase['mesh_name']
    package, prefix = get_mesh_package(mesh_name)
    add_config(config, package, '{}.cfg'.format(prefix), exception=True)
    if testcase['with_ice_shelf_cavities']:
        config.set('global_ocean', 'prefix', '{}wISC'.format(
            config.get('global_ocean', 'prefix')))

    name = testcase['name']
    add_config(config, 'compass.ocean.tests.global_ocean.{}'.format(name),
               '{}.cfg'.format(name), exception=False)

    add_descriptions_to_config(testcase, config)
