from compass.ocean.tests.buttermilk_bay.default import Default
from compass.testgroup import TestGroup


class ButtermilkBay(TestGroup):
    """
    A test group for Buttermilk Bay (subgrid wetting-and-drying) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='buttermilk_bay')
        for wetdry in ['standard', 'subgrid']:
            self.add_test_case(
                Default(test_group=self, wetdry=wetdry))
