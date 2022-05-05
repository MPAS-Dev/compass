from compass.testgroup import TestGroup
from compass.landice.tests.kangerlussuaq.mesh_gen import MeshGen


class Kangerlussuaq(TestGroup):
    """
    A test group for kangerlussuaq test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='kangerlussuaq')

        self.add_test_case(
            MeshGen(test_group=self))
