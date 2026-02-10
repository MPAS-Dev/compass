from compass.landice.tests.slm.circ_icesheet import CircIcesheetTest
from compass.testgroup import TestGroup


class Slm(TestGroup):
    """
    A test group for Sea-Level Model test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='slm')

        self.add_test_case(
            CircIcesheetTest(test_group=self))
