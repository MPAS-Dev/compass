from compass.landice.tests.isunnguata_sermia.mesh_gen import MeshGen
from compass.testgroup import TestGroup


class IsunnguataSermia(TestGroup):
    """
    A test group for isunnguata_sermia test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='isunnguata_sermia')

        self.add_test_case(
            MeshGen(test_group=self))
