from compass.testgroup import TestGroup
from compass.ocean.tests.turbulence_closure.decomp_test import DecompTest
from compass.ocean.tests.turbulence_closure.default import Default
from compass.ocean.tests.turbulence_closure.restart_test import RestartTest


class TurbulenceClosure(TestGroup):
    """
    A test group for turbulence closure test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='turbulence_closure')

        for resolution in [1e4]:
            self.add_test_case(
                DecompTest(test_group=self, resolution=resolution))
            self.add_test_case(
                RestartTest(test_group=self, resolution=resolution))
        for resolution in [1, 2, 1e4]:
            for forcing in ['cooling', 'evaporation']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution, forcing=forcing))


def configure(resolution, forcing, config):
    """
    Modify the configuration options for one of the turbulence closure test cases

    Parameters
    ----------
    resolution : float
        The resolution of the test case in meters

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    # The resolution parameters are different for different resolutions
    # to match existing simulations
    if resolution > 1e3:
        nx = 16
        ny = 50
        vert_levels = 20
        bottom_depth = 1e3
    elif resolution <= 1e3 and resolution > 5:
        nx = 50
        ny = 50
        vert_levels = 50
        bottom_depth = 100.0
    elif resolution <= 5 and resolution > 1:
        nx = 150
        ny = 150
        vert_levels = 50
        bottom_depth = 100.0
    elif resolution <= 1:
        nx = 128
        ny = 128
        vert_levels = 128
        bottom_depth = 128.0


    config.set('turbulence_closure', 'nx', f'{nx}')
    config.set('turbulence_closure', 'ny', f'{ny}')
    config.set('turbulence_closure', 'dc', f'{resolution}')
    config.set('vertical_grid', 'vert_levels', f'{vert_levels}')
    config.set('vertical_grid', 'bottom_depth', f'{bottom_depth}')

    if forcing == 'cooling':
        config.set('turbulence_closure', 'surface_heat_flux', '-100')
        config.set('turbulence_closure', 'surface_freshwater_flux', '0')
        config.set('turbulence_closure', 'interior_temperature_gradient', '0.1')
        config.set('turbulence_closure', 'interior_salinity_gradient', '0')
        config.set('turbulence_closure', 'wind_stress_zonal', '0')
    if forcing == 'evaporation':
        config.set('turbulence_closure', 'surface_heat_flux', '0')
        config.set('turbulence_closure', 'surface_freshwater_flux', '0.429')
        config.set('turbulence_closure', 'interior_temperature_gradient', '0')
        config.set('turbulence_closure', 'interior_salinity_gradient', '-0.025')
        config.set('turbulence_closure', 'wind_stress_zonal', '0')
