from compass.ocean.tests.drying_slope.decomp import Decomp
from compass.ocean.tests.drying_slope.convergence import Convergence
from compass.ocean.tests.drying_slope.default import Default
from compass.ocean.tests.drying_slope.loglaw import LogLaw
from compass.testgroup import TestGroup


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

        for method in ['standard', 'ramp']:
            for coord_type in ['sigma', 'single_layer']:
                for resolution in [0.25, 1.]:
                    self.add_test_case(
                        Default(test_group=self, resolution=resolution,
                                coord_type=coord_type, method=method))
                    self.add_test_case(
                        Decomp(test_group=self, resolution=resolution,
                               coord_type=coord_type, method=method))
            for coord_type in ['sigma']:
                self.add_test_case(
                    Convergence(test_group=self,
                                coord_type=coord_type,
                                method=method))
        for coord_type in ['sigma', 'single_layer']:
            for resolution in [0.25, 1.]:
                self.add_test_case(
                    LogLaw(test_group=self, resolution=resolution,
                           coord_type=coord_type, method='standard'))
