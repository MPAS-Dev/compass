from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.subdir import get_forward_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables, compare_timers
from compass.namelist import add_namelist_file
from compass.streams import add_streams_file
from compass.io import add_output_file


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
        description='performance and validation test')

    subdir = get_forward_sudbdir(mesh_name, initial_condition, with_bgc,
                                 time_integrator, name)
    set_testcase_subdir(testcase, subdir)

    step = add_step(testcase, forward, mesh_name=mesh_name,
                    with_ice_shelf_cavities=with_ice_shelf_cavities,
                    initial_condition=initial_condition, with_bgc=with_bgc,
                    time_integrator=time_integrator)

    if with_ice_shelf_cavities:
        module = __name__
        add_namelist_file(step, module, 'namelist.wisc')
        add_streams_file(step, module, 'streams.wisc')
        add_output_file(step, filename='land_ice_fluxes.nc')


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
    with_ice_shelf_cavities = testcase['with_ice_shelf_cavities']
    with_bgc = testcase['with_bgc']

    # get the these properties from the config options
    step = testcase['steps']['forward']
    for option in ['cores', 'min_cores', 'max_memory', 'max_disk', 'threads']:
        step[option] = config.getint('global_ocean',
                                     'forward_{}'.format(option))

    run_steps(testcase, test_suite, config, logger)

    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']

    if with_bgc:
        variables.extend(
            ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2',
             'ALK', 'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
             'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC', 'diatFe',
             'diatSi', 'diazChl', 'diazC', 'diazFe', 'phaeoChl', 'phaeoC',
             'phaeoFe'])

    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='forward/output.nc')

    if with_ice_shelf_cavities:
        variables = ['ssh', 'landIcePressure', 'landIceDraft',
                     'landIceFraction', 'landIceMask',
                     'landIceFrictionVelocity', 'topDrag', 'topDragMagnitude',
                     'landIceFreshwaterFlux', 'landIceHeatFlux',
                     'heatFluxToLandIce', 'landIceBoundaryLayerTemperature',
                     'landIceBoundaryLayerSalinity',
                     'landIceHeatTransferVelocity',
                     'landIceSaltTransferVelocity',
                     'landIceInterfaceTemperature', 'landIceInterfaceSalinity',
                     'accumulatedLandIceMass', 'accumulatedLandIceHeat']

        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='forward/land_ice_fluxes.nc')

    timers = ['time integration']
    compare_timers(timers, config, work_dir, rundir1='forward')
