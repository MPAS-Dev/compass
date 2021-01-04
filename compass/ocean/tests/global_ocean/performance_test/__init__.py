from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables, compare_timers


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
        description='performance and validation test')
    module = __name__

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)
    steps = dict()
    if with_ice_shelf_cavities:
        step = forward.collect(mesh_name, with_ice_shelf_cavities, with_bgc,
                               time_integrator, cores=4, threads=1,
                               testcase_module=module,
                               namelist_file='namelist.wisc',
                               streams_file='streams.wisc')
    else:
        step = forward.collect(mesh_name, with_ice_shelf_cavities, with_bgc,
                               time_integrator, cores=4, threads=1)
    steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name
    testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities
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
    steps = ['forward']
    work_dir = testcase['work_dir']
    with_ice_shelf_cavities = testcase['with_ice_shelf_cavities']
    with_bgc = testcase['with_bgc']
    run_steps(testcase, test_suite, config, steps, logger)

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
