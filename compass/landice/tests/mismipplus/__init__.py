from compass.landice.tests.mismipplus.mesh_gen import MeshGen
from compass.landice.tests.mismipplus.smoke_test import SmokeTest
from compass.testgroup import TestGroup


class MISMIPplus(TestGroup):
    """
    A test group for MISMIP+ test cases.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='mismipplus')

        self.add_test_case(SmokeTest(test_group=self))

        # Test case for setting up mesh and I.C.'s for MISMIP
        self.add_test_case(MeshGen(test_group=self))
