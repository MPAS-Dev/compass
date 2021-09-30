from compass.testgroup import TestGroup

from compass.ocean.tests.isomip_plus.ocean_test import OceanTest


class IsomipPlus(TestGroup):
    """
    A test group for ice-shelf 2D test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='isomip_plus')

        for resolution in [2., 5.]:
            for experiment in ['Ocean0', 'Ocean1', 'Ocean2']:
                for vertical_coordinate in ['z-star']:
                    self.add_test_case(
                        OceanTest(test_group=self, resolution=resolution,
                                  experiment=experiment,
                                  vertical_coordinate=vertical_coordinate))
            for experiment in ['Ocean0']:
                for vertical_coordinate in ['z-star']:
                    self.add_test_case(
                        OceanTest(test_group=self, resolution=resolution,
                                  experiment=experiment,
                                  vertical_coordinate=vertical_coordinate,
                                  time_varying_forcing=True))
