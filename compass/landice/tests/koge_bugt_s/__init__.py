from compass.testgroup import TestGroup
from compass.landice.tests.koge_bugt_s.mesh_gen import MeshGen


class KogeBugtS(TestGroup):
    """
    A test group for koge_bugt_s test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='koge_bugt_s')

        self.add_test_case(
            MeshGen(test_group=self))
