from compass.testgroup import TestGroup
from compass.landice.tests.humboldt.default import Default


class Humboldt(TestGroup):
    """
    A test group for humboldt test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='humboldt')

        self.add_test_case(
            Default(test_group=self))
