from compass.testgroup import TestGroup
from compass.ocean.tests.soma.default import Default


class Soma(TestGroup):
    """
    A test group for Simulating Ocean Mesoscale Activity (SOMA) test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='soma')

        for resolution in ['4km', '8km', '16km', '32km']:
            self.add_test_case(
                Default(test_group=self, resolution=resolution))
