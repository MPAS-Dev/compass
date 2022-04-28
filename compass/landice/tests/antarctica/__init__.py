from compass.testgroup import TestGroup
from compass.landice.tests.antarctica.mesh_gen import MeshGen


class Antarctica(TestGroup):
    """
    A test group for Antarctica test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='antarctica')

        self.add_test_case(
            MeshGen(test_group=self))
