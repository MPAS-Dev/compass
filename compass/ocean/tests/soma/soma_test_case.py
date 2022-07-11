from compass.testcase import TestCase
from compass.ocean.tests.soma.base_mesh import BaseMesh
from compass.ocean.tests.soma.culled_mesh import CulledMesh
from compass.ocean.tests.soma.initial_state import InitialState
from compass.ocean.tests.soma.forward import Forward
from compass.ocean.tests.soma.analysis import Analysis
from compass.validate import compare_variables, compare_timers


class SomaTestCase(TestCase):
    """
    The class for all test cases in the SOMA test group.  The test case creates
    the mesh and initial condition, then performs a forward run with analysis
    members.  An analysis step is included if the simulation includes
    particles.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_particles : bool
        Whether particles are include in the simulation

    with_surface_restoring : bool
        Whether surface restoring is included in the simulation

    long : bool
        Whether to run a long (3-year) simulation to quasi-equilibrium

    three_layer : bool
        Whether to use only 3 vertical layers and no continental shelf
    """

    def __init__(self, test_group, resolution, with_particles,
                 with_surface_restoring, long, three_layer):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.soma.Soma
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        with_particles : bool
            Whether particles are include in the simulation

        with_surface_restoring : bool
            Whether surface restoring is included in the simulation

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium

        three_layer : bool
            Whether to use only 3 vertical layers and no continental shelf
        """
        self.resolution = resolution
        self.with_particles = with_particles
        self.with_surface_restoring = with_surface_restoring
        self.long = long
        self.three_layer = three_layer

        name = None
        if three_layer:
            name = 'three_layer'
        if long:
            if name is None:
                name = 'long'
            else:
                name = f'{name}_long'

        if with_surface_restoring:
            if name is None:
                name = 'surface_restoring'
            else:
                name = f'{name}_with_surface_restoring'

        if with_particles:
            if name is None:
                name = 'particles'
            else:
                name = f'{name}_with_particles'

        if name is None:
            name = 'default'

        subdir = f'{resolution}/{name}'

        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(BaseMesh(
            test_case=self, resolution=resolution))

        self.add_step(InitialState(
            test_case=self, resolution=resolution,
            with_surface_restoring=with_surface_restoring,
            three_layer=three_layer, mark_land=True))

        self.add_step(CulledMesh(
            test_case=self))

        self.add_step(InitialState(
            test_case=self, resolution=resolution,
            with_surface_restoring=with_surface_restoring,
            three_layer=three_layer, mark_land=False))

        options = dict()
        if with_surface_restoring:
            options['config_use_activeTracers_surface_restoring'] = '.true.'

        self.add_step(Forward(
            test_case=self, resolution=resolution,
            with_particles=with_particles,
            with_surface_restoring=with_surface_restoring, long=long,
            three_layer=three_layer))

        if with_particles:
            self.add_step(Analysis(test_case=self, resolution=resolution))

    def configure(self):
        """
        Set config options that are different from the defaults
        """
        if self.three_layer:
            config = self.config
            # remove the continental shelf
            config.set('soma', 'phi', '1e-16')
            config.set('soma', 'shelf_depth', '0.0')

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

        if self.with_particles:
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
