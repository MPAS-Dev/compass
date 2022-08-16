from compass.testcase import TestCase
from compass.landice.tests.hydro_radial.setup_mesh import SetupMesh
from compass.landice.tests.hydro_radial.run_model import RunModel
from compass.landice.tests.hydro_radial.visualize import Visualize


class SpinupTest(TestCase):
    """
    A spin-up test case for the radially symmetric hydrological test group that
    creates the mesh and initial condition, then performs a long short forward
    run on 4 cores until a quasi-steady state is reached.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.hydro_radial.HydroRadial
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='spinup_test')

        self.add_step(
            SetupMesh(test_case=self, initial_condition='zero'))
        step = RunModel(test_case=self, ntasks=4, openmp_threads=1)
        step.add_namelist_file(
            'compass.landice.tests.hydro_radial.spinup_test',
            'namelist.landice')

        step.add_streams_file(
            'compass.landice.tests.hydro_radial.spinup_test',
            'streams.landice')
        self.add_step(step)

        step = Visualize(test_case=self)
        self.add_step(step, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed because we're doing the default: running all
    # steps
