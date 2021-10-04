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
                Default(test_group=self, resolution=resolution))
            self.add_test_case(
                RestartTest(test_group=self, resolution=resolution))


def configure(resolution, config):
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
                  '4km': {'nx': 40,
                          'ny': 126,
                          'dc': 4e3},
                  '1km': {'nx': 160,
                          'ny': 500,
                          'dc': 1e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('turbulence_closure', param, '{}'.format(res_params[param]))
