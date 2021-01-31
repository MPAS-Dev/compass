from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.ice_shelf_2d import initial_state, ssh_adjustment, \
    forward
from compass.ocean.tests import ice_shelf_2d
from compass.validate import compare_variables
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
    resolution = testcase['resolution']
    testcase['description'] = \
        '2D ice-shelf {} restart test with frazil'.format(resolution)

    set_testcase_subdir(testcase, '{}/{}'.format(resolution, testcase['name']))

    add_step(testcase, initial_state, resolution=resolution)
    add_step(testcase, ssh_adjustment, resolution=resolution, cores=4,
             threads=1)

    module = __name__
    for part in ['full', 'restart']:
        name = '{}_run'.format(part)
        step = add_step(testcase, forward, name=name, subdir=name,
                        resolution=resolution, cores=4, threads=1,
                        with_frazil=True)
        add_namelist_file(step, module, 'namelist.{}'.format(part))
        add_streams_file(step, module, 'streams.{}'.format(part))


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
    ice_shelf_2d.configure(testcase, config)


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

        variables = ['ssh', 'landIcePressure', 'landIceDraft',
                     'landIceFraction',
                     'landIceMask', 'landIceFrictionVelocity', 'topDrag',
                     'topDragMagnitude', 'landIceFreshwaterFlux',
                     'landIceHeatFlux', 'heatFluxToLandIce',
                     'landIceBoundaryLayerTemperature',
                     'landIceBoundaryLayerSalinity',
                     'landIceHeatTransferVelocity',
                     'landIceSaltTransferVelocity',
                     'landIceInterfaceTemperature',
                     'landIceInterfaceSalinity', 'accumulatedLandIceMass',
                     'accumulatedLandIceHeat']
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/land_ice_fluxes.nc',
                          filename2='restart_run/land_ice_fluxes.nc')

        variables = ['accumulatedFrazilIceMass',
                     'accumulatedFrazilIceSalinity',
                     'seaIceEnergy', 'frazilLayerThicknessTendency',
                     'frazilTemperatureTendency', 'frazilSalinityTendency',
                     'frazilSurfacePressure', 'accumulatedLandIceFrazilMass']
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='full_run/frazil.nc',
                          filename2='restart_run/frazil.nc')
