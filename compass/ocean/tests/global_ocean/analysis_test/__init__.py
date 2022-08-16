import traceback
import os

from compass.validate import compare_variables, compare_timers
from compass.ocean.tests.global_ocean.forward import ForwardTestCase, \
    ForwardStep


class AnalysisTest(ForwardTestCase):
    """
    A test case for performing a short forward run with an MPAS-Ocean global
    initial condition and check nearly all MPAS-Ocean analysis members to make
    sure they run successfully and output is identical to a baseline (if one
    is provided).

    Attributes
    ----------
    variables : dict
        A dictionary of output files and the variables in each to validate
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run
        """
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='analysis_test')

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, ntasks=4,
                           openmp_threads=1)

        self.variables = {
            'output.nc':
                ['temperature', 'salinity', 'layerThickness',
                 'normalVelocity'],
            'analysis_members/globalStats.0001-01-01_00.00.00.nc':
                ['kineticEnergyCellMax', 'kineticEnergyCellMin',
                 'kineticEnergyCellAvg', 'temperatureAvg', 'salinityAvg'],
            'analysis_members/debugDiagnostics.0001-01-01.nc':
                ['rx1MaxCell'],
            'analysis_members/highFrequencyOutput.0001-01-01.nc':
                ['temperatureAt250m'],
            'analysis_members/mixedLayerDepths.0001-01-01.nc':
                ['dThreshMLD', 'tThreshMLD'],
            'analysis_members/waterMassCensus.0001-01-01_00.00.00.nc':
                ['waterMassCensusTemperatureValues'],
            'analysis_members/eliassenPalm.0001-01-01.nc':
                ['EPFT'],
            'analysis_members/'
            'layerVolumeWeightedAverage.0001-01-01_00.00.00.nc':
                ['avgVolumeTemperature', 'avgVolumeRelativeVorticityCell'],
            'analysis_members/okuboWeiss.0001-01-01_00.00.00.nc':
                ['okuboWeiss'],
            'analysis_members/zonalMeans.0001-01-01_00.00.00.nc':
                ['velocityZonalZonalMean', 'temperatureZonalMean'],
            'analysis_members/'
            'meridionalHeatTransport.0001-01-01_00.00.00.nc':
                ['meridionalHeatTransportLat'],
            'analysis_members/'
            'surfaceAreaWeightedAverages.0001-01-01_00.00.00.nc':
                ['avgSurfaceSalinity', 'avgSeaSurfacePressure'],
            'analysis_members/'
            'eddyProductVariables.0001-01-01.nc':
                ['SSHSquared', 'velocityZonalSquared',
                 'velocityZonalTimesTemperature'],
            'analysis_members/oceanHeatContent.0001-01-01.nc':
                ['oceanHeatContentSfcToBot', 'oceanHeatContentSfcTo700m',
                 'oceanHeatContent700mTo2000m', 'oceanHeatContent2000mToBot'],
            'analysis_members/mixedLayerHeatBudget.0001-01-01.nc':
                ['temperatureHorAdvectionMLTend', 'salinityHorAdvectionMLTend',
                 'temperatureML', 'salinityML', 'bruntVaisalaFreqML']}

        for output in self.variables:
            step.add_output_file(output)

        module = self.__module__
        step.add_namelist_file(module, 'namelist.forward')
        step.add_streams_file(module, 'streams.forward')
        self.add_step(step)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        config = self.config
        work_dir = self.work_dir

        failed = list()
        for filename, variables in self.variables.items():
            try:
                compare_variables(test_case=self, variables=variables,
                                  filename1=os.path.join('forward', filename))
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
                  'write_surfaceAreaWeightedAverages',
                  'compute_waterMassCensus',
                  'write_waterMassCensus', 'compute_zonalMean',
                  'write_zonalMean',
                  'compute_eddyProductVariables', 'write_eddyProductVariables',
                  'compute_oceanHeatContent', 'write_oceanHeatContent',
                  'compute_mixedLayerHeatBudget', 'write_mixedLayerHeatBudget']
        compare_timers(self, timers, rundir1='forward')

        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        if self.init.with_bgc:
            variables.extend(
                ['PO4', 'NO3', 'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2',
                 'ALK', 'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
                 'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC',
                 'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazFe', 'phaeoChl',
                 'phaeoC', 'phaeoFe'])

        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

        timers = ['time integration']
        compare_timers(self, timers, rundir1='forward')
