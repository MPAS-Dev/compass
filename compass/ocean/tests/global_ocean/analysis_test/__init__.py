import traceback

from compass.testcase import run_steps, get_testcase_default
from compass.ocean.tests.global_ocean import forward
from compass.ocean.tests import global_ocean
from compass.validate import compare_variables, compare_timers


def collect(mesh_name, time_integrator):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the run

    Returns
    -------
    testcase : dict
        A dict of properties of this test case, including its steps
    """
    description = 'global ocean {} - {} analysis test'.format(
        mesh_name, time_integrator)
    module = __name__

    name = module.split('.')[-1]
    subdir = '{}/{}/{}'.format(mesh_name, name, time_integrator)
    steps = dict()
    step = forward.collect(mesh_name=mesh_name, cores=4, threads=1,
                           testcase_module=module,
                           namelist_file='namelist.forward',
                           streams_file='streams.forward',
                           time_integrator=time_integrator)
    steps[step['name']] = step

    testcase = get_testcase_default(module, description, steps, subdir=subdir)
    testcase['mesh_name'] = mesh_name

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
    run_steps(testcase, test_suite, config, steps, logger)

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
