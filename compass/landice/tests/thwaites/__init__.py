from compass.testgroup import TestGroup
from compass.landice.tests.thwaites.decomposition_test import DecompositionTest
from compass.landice.tests.thwaites.restart_test import RestartTest
from compass.landice.tests.thwaites.high_res_mesh import HighResMesh

class Thwaites(TestGroup):
    """
    A test group for low-res 4-14km Thwaites test cases.
    This test group uses a pre-made mesh file.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='thwaites')

        self.add_test_case(DecompositionTest(test_group=self))

        self.add_test_case(RestartTest(test_group=self))

        self.add_test_case(HighResMesh(test_group=self))
