from compass.testgroup import TestGroup
from compass.ocean.tests.ziso.with_frazil import WithFrazil
from compass.config import add_config
from compass.ocean.tests.ziso.ziso_test_case import ZisoTestCase


class Ziso(TestGroup):
    """
    A test group for Zonally Invariant Southern Ocean (ZISO) test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ziso')

        for resolution in ['2.5km', '5km', '10km', '20km']:
            self.add_test_case(
                ZisoTestCase(test_group=self, resolution=resolution,
                             with_particles=False, long=False))
            self.add_test_case(
                ZisoTestCase(test_group=self, resolution=resolution,
                             with_particles=False, long=True))
            self.add_test_case(
                ZisoTestCase(test_group=self, resolution=resolution,
                             with_particles=True, long=False))
        for resolution in ['20km']:
            self.add_test_case(
                WithFrazil(test_group=self, resolution=resolution))


def configure(name, resolution, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    name : str
        the name of the test case

    resolution : str
        The resolution of the test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    res_params = {'20km': {'nx': 50,
                           'ny': 112,
                           'dc': 20e3},
                  '10km': {'nx': 100,
                           'ny': 228,
                           'dc': 10e3},
                  '5km': {'nx': 200,
                          'ny': 460,
                          'dc': 5e3},
                  '2.5km': {'nx': 400,
                            'ny': 922,
                            'dc': 2.5e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('ziso', param, '{}'.format(res_params[param]))
