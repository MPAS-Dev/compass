from compass.ocean.tests.dam_break.default import Default
from compass.ocean.tests.dam_break.ramp import Ramp
from compass.testgroup import TestGroup


class DamBreak(TestGroup):
    """
    A test group for dam break (wetting-and-drying) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='dam_break')

        for resolution in [0.04, 0.12]:
            for use_lts in [True, False]:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            use_lts=use_lts))
                self.add_test_case(
                    Ramp(test_group=self, resolution=resolution,
                         use_lts=use_lts))
