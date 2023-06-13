from compass.ocean.tests.overflow.initial_state import InitialState
from compass.ocean.tests.overflow.nonhydro.forward import Forward
from compass.testcase import TestCase


class Nonhydro(TestCase):
    """
    The nonhydro test case for the overflow test group creates the mesh
    and initial condition, then performs a forward runs with the
    nonhydrostatic version of MPAS-O.

    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.nonhydro.Nonhydro
            The test group that this test case belongs to
        """
        name = 'nonhydro'
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            InitialState(test_case=self))
        self.add_step(
            Forward(test_case=self, name='forward'))

    # no run() is needed because we're doing the default: running all steps
