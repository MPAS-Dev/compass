from compass.landice.tests.crane.mesh_gen import MeshGen
from compass.testgroup import TestGroup


class Crane(TestGroup):
    """
    A test group for Crane Glacier test cases.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='crane')

        self.add_test_case(MeshGen(test_group=self))
