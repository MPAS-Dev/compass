from compass.ocean.tests.utility.extrap_woa import ExtrapWoa
from compass.testgroup import TestGroup


class Utility(TestGroup):
    """
    A test group for general ocean utilities
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='utility')

        self.add_test_case(ExtrapWoa(test_group=self))
