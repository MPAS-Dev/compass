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

        for resolution in ['10km']:
            self.add_test_case(
                DecompTest(test_group=self, resolution=resolution))
            self.add_test_case(
                RestartTest(test_group=self, resolution=resolution))
        for resolution in ['1m', '2m', '10km']:
            for forcing in ['cooling', 'evaporation']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution, forcing=forcing))


def configure(resolution, forcing, config):
    """
    Modify the configuration options for one of the turbulence closure test cases

    Parameters
    ----------
    resolution : str
        The resolution of the test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    res_params = {'10km': {'nx': 16,
                           'ny': 50,
                           'dc': 10e3},
                  '2m': {'nx': 150,
                         'ny': 150,
                         'dc': 2},
                  '1m': {'nx': 128,
                         'ny': 128,
                         'dc': 1}}
    vert_params = {'10km': {'vert_levels': 20,
                            'bottom_depth': 1e3},
                   '2m': {'vert_levels': 50,
                          'bottom_depth': 100.0},
                   '1m': {'vert_levels': 128,
                          'bottom_depth': 128.0}}

    if resolution not in res_params:
        raise ValueError(f'Unsupported resolution {resolution}. '
                         f'Supported values are: {list(res_params)}')

    res_params = res_params[resolution]
    for param in res_params:
        config.set('turbulence_closure', param, f'{res_params[param]}')
    vert_params = vert_params[resolution]
    for param in vert_params:
        config.set('vertical_grid', param, f'{vert_params[param]}')

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
