from compass.ocean.tests.overflow.hydro_vs_nonhydro.forward import Forward
from compass.ocean.tests.overflow.hydro_vs_nonhydro.visualize import Visualize
from compass.ocean.tests.overflow.initial_state import InitialState
from compass.testcase import TestCase


class HydroVsNonhydro(TestCase):
    """
    The hydro vs nonhydro test case for the overflow group creates the
    mesh and initial condition, then performs two forward runs, one with
    the standard hydrostatic version of MPAS-O, and the second with the
    nonhydrostatic version. Finally, it produces a plot comparing the
    temperature profiles for the two models.

    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.hydro_vs_nonhydro.HydroVSNonhydro
            The test group that this test case belongs to
        """
        name = 'hydro_vs_nonhydro'
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            InitialState(test_case=self))
        self.add_step(
            Forward(test_case=self, nonhydro_mode=False, name='hydro'))
        self.add_step(
            Forward(test_case=self, nonhydro_mode=True, name='nonhydro'))
        self.add_step(
            Visualize(test_case=self))

    # no run() is needed because we're doing the default: running all steps
