from compass.ocean.tests.isomip_plus.isomip_plus_test import IsomipPlusTest
from compass.testgroup import TestGroup


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
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate))

            self.add_test_case(
                IsomipPlusTest(
                    test_group=self, resolution=resolution,
                    experiment='Ocean0',
                    vertical_coordinate='sigma'))

            for experiment in ['Ocean0']:
                for vertical_coordinate in ['z-star']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            time_varying_forcing=True))
                for vertical_coordinate in ['z-star', 'sigma', 'single_layer']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            thin_film_present=True))
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            time_varying_forcing=True,
                            thin_film_present=True))
                for vertical_coordinate in ['sigma', 'single_layer']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            time_varying_forcing=True,
                            time_varying_load='increasing',
                            thin_film_present=True))
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            time_varying_forcing=True,
                            time_varying_load='decreasing',
                            thin_film_present=True))
                for vertical_coordinate in ['sigma']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            tidal_forcing=True,
                            thin_film_present=True))
                for vertical_coordinate in ['single_layer']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            tidal_forcing=True,
                            thin_film_present=False))
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            tidal_forcing=True,
                            thin_film_present=True))

        for resolution in [2.]:
            for experiment in ['Ocean0', 'Ocean1', 'Ocean2']:
                for vertical_coordinate in ['z-star']:
                    self.add_test_case(
                        IsomipPlusTest(
                            test_group=self, resolution=resolution,
                            experiment=experiment,
                            vertical_coordinate=vertical_coordinate,
                            planar=False))
