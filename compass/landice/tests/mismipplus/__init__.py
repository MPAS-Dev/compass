from compass.testgroup import TestGroup
from compass.landice.tests.mismipplus.dt_convergence_test import DtConvergenceTest


class MISMIPplus(TestGroup):
    """
    A test group for MISMIP+ test cases.
    This test group uses a pre-made mesh file.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='mismipplus')

        self.add_test_case(DtConvergenceTest(test_group=self))
