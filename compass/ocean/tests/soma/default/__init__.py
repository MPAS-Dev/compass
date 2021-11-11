from compass.testcase import TestCase
from compass.ocean.tests.soma.initial_state import InitialState
from compass.ocean.tests.soma.forward import Forward
from compass.ocean.tests.soma.default.analysis import Analysis
from compass.validate import compare_variables, compare_timers


class Default(TestCase):
    """
    The default test case for the SOMA test group simply creates the mesh and
    initial condition, then performs a short forward run with analysis members.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.soma.Soma
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'default'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        step = InitialState(test_case=self, resolution=resolution)
        # add the namelist options specific ot this test case
        package = 'compass.ocean.tests.soma.default'
        for out_name in ['namelist_mark_land.ocean', 'namelist.ocean']:
            step.add_namelist_file(package, 'namelist.init', mode='init',
                                   out_name=out_name)
        self.add_step(step)

        with_particles = self.resolution == '32km'
        self.add_step(
            Forward(test_case=self, resolution=resolution,
                    with_particles=with_particles))

        if resolution == '32km':
            self.add_step(Analysis(test_case=self, resolution=resolution))

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        """
        variables = ['bottomDepth', 'layerThickness', 'maxLevelCell',
                     'temperature', 'salinity']
        compare_variables(
            test_case=self, variables=variables,
            filename1='initial_state/initial_state.nc')

        variables = ['temperature', 'layerThickness']
        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/output/output.0001-01-01_00.00.00.nc')

        if self.resolution == '32km':
            # just do particle validation at coarse res
            variables = [
                'xParticle', 'yParticle', 'zParticle', 'zLevelParticle',
                'buoyancyParticle', 'indexToParticleID', 'currentCell',
                'transfered', 'numTimesReset']
            compare_variables(test_case=self, variables=variables,
                              filename1='forward/analysis_members/'
                                        'lagrPartTrack.0001-01-01_00.00.00.nc')

            timers = ['init_lagrPartTrack', 'compute_lagrPartTrack',
                      'write_lagrPartTrack', 'restart_lagrPartTrack',
                      'finalize_lagrPartTrack']
            compare_timers(self, timers, rundir1='forward')
