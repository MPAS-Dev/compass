from compass.ocean.tests.lock_exchange.forward import Forward
from compass.ocean.tests.lock_exchange.initial_state import InitialState
from compass.ocean.tests.lock_exchange.visualize import Visualize
from compass.testcase import TestCase


class Hydro(TestCase):
    """
    The default test case for the solitary wave test simply creates the
    mesh and initial condition, then performs two forward runs, one with
    the standard hydrostatic version of MPAS-O, and the second with the
    nonhydrostatic version.

    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.nonhydro.Nonhydro
            The test group that this test case belongs to
        """
        name = 'hydro'
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            InitialState(test_case=self))
        self.add_step(
            Forward(test_case=self, name='forward'))
        self.add_step(
            Visualize(test_case=self))

    # no run() is needed because we're doing the default: running all steps
