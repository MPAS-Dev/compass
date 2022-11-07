from compass.testgroup import TestGroup
from compass.ocean.tests.merry_go_round.default import Default
from compass.ocean.tests.merry_go_round.convergence_test import ConvergenceTest

class MerryGoRound(TestGroup):
    """
    A test group for tracer advection test cases "merry-go-round"
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='merry_go_round')

        self.add_test_case(Default(test_group=self))
        self.add_test_case(ConvergenceTest(test_group=self))
