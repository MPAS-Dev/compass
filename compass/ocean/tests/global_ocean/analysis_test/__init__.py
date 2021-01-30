import traceback

from compass.testcase import set_testcase_subdir, add_step, run_steps
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables, compare_timers
from compass.ocean.tests.global_ocean.description import get_description
from compass.ocean.tests.global_ocean.init import get_init_sudbdir
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
        description='analysis test')

    init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)
    subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)
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

    variables = {
        'forward/output.nc':
            ['temperature', 'salinity', 'layerThickness', 'normalVelocity'],
        'forward/analysis_members/globalStats.0001-01-01_00.00.00.nc':
            ['kineticEnergyCellMax', 'kineticEnergyCellMin',
             'kineticEnergyCellAvg', 'temperatureAvg', 'salinityAvg'],
        'forward/analysis_members/debugDiagnostics.0001-01-01.nc':
            ['rx1MaxCell'],
        'forward/analysis_members/highFrequencyOutput.0001-01-01.nc':
            ['temperatureAt250m'],
        'forward/analysis_members/mixedLayerDepths.0001-01-01.nc':
            ['dThreshMLD', 'tThreshMLD'],
        'forward/analysis_members/waterMassCensus.0001-01-01_00.00.00.nc':
            ['waterMassCensusTemperatureValues'],
        'forward/analysis_members/eliassenPalm.0001-01-01.nc':
            ['EPFT'],
        'forward/analysis_members/'
        'layerVolumeWeightedAverage.0001-01-01_00.00.00.nc':
            ['avgVolumeTemperature', 'avgVolumeRelativeVorticityCell'],
        'forward/analysis_members/okuboWeiss.0001-01-01_00.00.00.nc':
            ['okuboWeiss'],
        'forward/analysis_members/zonalMeans.0001-01-01_00.00.00.nc':
            ['velocityZonalZonalMean', 'temperatureZonalMean'],
        'forward/analysis_members/'
        'meridionalHeatTransport.0001-01-01_00.00.00.nc':
            ['meridionalHeatTransportLat'],
        'forward/analysis_members/'
        'surfaceAreaWeightedAverages.0001-01-01_00.00.00.nc':
            ['avgSurfaceSalinity', 'avgSeaSurfacePressure'],
        'forward/analysis_members/'
        'eddyProductVariables.0001-01-01.nc':
            ['SSHSquared', 'velocityZonalSquared',
             'velocityZonalTimesTemperature'],
        'forward/analysis_members/oceanHeatContent.0001-01-01.nc':
            ['oceanHeatContentSfcToBot', 'oceanHeatContentSfcTo700m',
             'oceanHeatContent700mTo2000m', 'oceanHeatContent2000mToBot'],
        'forward/analysis_members/mixedLayerHeatBudget.0001-01-01.nc':
            ['temperatureHorAdvectionMLTend', 'salinityHorAdvectionMLTend',
             'temperatureML', 'salinityML', 'bruntVaisalaFreqML']}

    failed = list()
    for filename, variables in variables.items():
        try:
            compare_variables(variables, config, work_dir=work_dir,
                              filename1=filename)
        except ValueError:
            traceback.print_exc()
            failed.append(filename)

    if len(failed) > 0:
        raise ValueError('Comparison failed, see above, for the following '
                         'files:\n{}.'.format('\n'.join(failed)))

    timers = ['compute_globalStats', 'write_globalStats',
              'compute_debugDiagnostics', 'write_debugDiagnostics',
              'compute_eliassenPalm', 'write_eliassenPalm',
              'compute_highFrequency', 'write_highFrequency',
              'compute_layerVolumeWeightedAverage',
              'write_layerVolumeWeightedAverage',
              'compute_meridionalHeatTransport',
              'write_meridionalHeatTransport', 'compute_mixedLayerDepths',
              'write_mixedLayerDepths', 'compute_okuboWeiss',
              'write_okuboWeiss', 'compute_surfaceAreaWeightedAverages',
              'write_surfaceAreaWeightedAverages', 'compute_waterMassCensus',
              'write_waterMassCensus', 'compute_zonalMean', 'write_zonalMean',
              'compute_eddyProductVariables', 'write_eddyProductVariables',
              'compute_oceanHeatContent', 'write_oceanHeatContent',
              'compute_mixedLayerHeatBudget', 'write_mixedLayerHeatBudget']
    compare_timers(timers, config, work_dir, rundir1='forward')
