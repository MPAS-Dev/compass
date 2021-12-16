from compass.testcase import TestCase
from compass.ocean.tests.ziso.initial_state import InitialState
from compass.ocean.tests.ziso.forward import Forward
from compass.ocean.tests import ziso
from compass.validate import compare_variables, compare_timers


class ZisoTestCase(TestCase):
    """
    The default test case for the ZISO test group simply creates the mesh and
    initial condition, then performs a short forward run with analysis members
    but without frazil.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_particles : bool
        Whether particles are include in the simulation

    long : bool
        Whether to run a long (3-year) simulation to quasi-equilibrium
    """

    def __init__(self, test_group, resolution, with_particles, long):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.ziso.Ziso
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        with_particles : bool
            Whether particles are include in the simulation

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium
        """
        name = 'default'
        self.resolution = resolution
        self.with_particles = with_particles
        self.long = long
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        res_params = {'20km': {'cores': 4, 'min_cores': 2}}

        if resolution not in res_params:
            raise ValueError(
                'Unsupported resolution {}. Supported values are: '
                '{}'.format(resolution, list(res_params)))

        res_params = res_params[resolution]

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         with_frazil=False))
        step = Forward(test_case=self, resolution=resolution,
                       cores=res_params['cores'],
                       min_cores=res_params['min_cores'],
                       with_analysis=True, with_frazil=False, long=long)

        if with_particles:
            # particles are on only for the 20km test case
            step.add_namelist_file('compass.ocean.tests.ziso.default',
                                   'namelist.particles')
        self.add_step(step)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ziso.configure(self.name, self.resolution, self.config)

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        config = self.config
        work_dir = self.work_dir

        variables = ['bottomDepth', 'layerThickness', 'maxLevelCell',
                     'temperature', 'salinity']
        compare_variables(
            test_case=self, variables=variables,
            filename1='initial_state/ocean.nc')

        variables = ['temperature', 'layerThickness']
        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/output/output.0001-01-01_00.00.00.nc')

        if self.with_particles:
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
