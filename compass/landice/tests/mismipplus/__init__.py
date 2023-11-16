from compass.landice.tests.mismipplus.makemeshtest import MakeMeshTest
from compass.landice.tests.mismipplus.smoke_test import SmokeTest
from compass.testgroup import TestGroup


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

        self.add_test_case(SmokeTest(test_group=self))

        # Dummy test case for testing the mesh stepup.
        self.add_test_case(MakeMeshTest(test_group=self))
