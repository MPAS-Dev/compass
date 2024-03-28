from compass.ocean.tests.baroclinic_gyre.gyre_test_case import GyreTestCase
from compass.testgroup import TestGroup


class BaroclinicGyre(TestGroup):
    """
    A test group for baroclinic gyre test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='baroclinic_gyre')

        for resolution in [20000., 80000.]:
            self.add_test_case(
                GyreTestCase(test_group=self, resolution=resolution,
                             long=False))
            self.add_test_case(
                GyreTestCase(test_group=self, resolution=resolution,
                             long=True))
