from compass.testgroup import TestGroup
from compass.ocean.tests.nonhydro.solitary_wave import SolitaryWave

class Nonhydro(TestGroup):
    """
    A test group for nonhydrostatic test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='nonhydro')
     
        self.add_test_case(
            SolitaryWave(test_group=self))

