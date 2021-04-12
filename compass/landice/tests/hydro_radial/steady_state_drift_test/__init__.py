from compass.testcase import TestCase
from compass.landice.tests.hydro_radial.setup_mesh import SetupMesh
from compass.landice.tests.hydro_radial.run_model import RunModel
from compass.landice.tests.hydro_radial.visualize import Visualize


class SteadyStateDriftTest(TestCase):
    """
    This test case assesses the drift of the model away from an initial
    condition that is a quasi-exact solution.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.hydro_radial.HydroRadial
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='steady_state_drift_test')

        SetupMesh(test_case=self, initial_condition='exact')
        RunModel(test_case=self, cores=4, threads=1)
        Visualize(test_case=self, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed because we're doing the default: running all
    # steps
