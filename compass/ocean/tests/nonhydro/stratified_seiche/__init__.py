from compass.testcase import TestCase
from compass.ocean.tests.nonhydro.stratified_seiche.initial_state import \
    InitialState
from compass.ocean.tests.nonhydro.stratified_seiche.forward import Forward
from compass.ocean.tests.nonhydro.stratified_seiche.visualize import Visualize


class StratifiedSeiche(TestCase):
    """
    The default test case for the stratified seiche test simply creates the
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
        name = 'stratified_seiche'
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
