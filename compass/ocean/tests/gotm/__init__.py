from compass.testgroup import TestGroup
from compass.ocean.tests.gotm.default import Default


class Gotm(TestGroup):
    """
    A test group for General Ocean Turbulence Model (GOTM) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='gotm')

        self.add_test_case(Default(test_group=self))
