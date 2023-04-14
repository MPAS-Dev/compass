from compass.landice.tests.issunguata_sermia.mesh_gen import MeshGen
from compass.testgroup import TestGroup


class IssunguataSermia(TestGroup):
    """
    A test group for issunguata_sermia test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='issunguata_sermia')

        self.add_test_case(
            MeshGen(test_group=self))
