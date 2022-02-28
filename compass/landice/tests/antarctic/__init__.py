from compass.testgroup import TestGroup
from compass.landice.tests.antarctic.default import Default


class Antarctic(TestGroup):
    """
    A test group for antarctic test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='antarctic')

        self.add_test_case(
            Default(test_group=self))
