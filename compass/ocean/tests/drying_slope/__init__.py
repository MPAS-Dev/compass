from compass.testgroup import TestGroup
from compass.ocean.tests.drying_slope.default import Default


class DryingSlope(TestGroup):
    """
    A test group for drying slope (wetting-and-drying) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='drying_slope')

        for resolution in ['1km']:
            for coord_type in ['sigma']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            coord_type=coord_type))