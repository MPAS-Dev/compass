from compass.testgroup import TestGroup
from compass.ocean.tests.baroclinic_channel.decomp_test import DecompTest
from compass.ocean.tests.baroclinic_channel.default import Default
from compass.ocean.tests.baroclinic_channel.restart_test import RestartTest
from compass.ocean.tests.baroclinic_channel.rpe_test import RpeTest
from compass.ocean.tests.baroclinic_channel.threads_test import ThreadsTest


class BaroclinicChannel(TestGroup):
    """
    A test group for baroclinic channel test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='baroclinic_channel')

        for resolution in ['1km', '4km', '10km']:
            self.add_test_case(
                RpeTest(test_group=self, resolution=resolution))
        for resolution in ['10km']:
            self.add_test_case(
                DecompTest(test_group=self, resolution=resolution))
            self.add_test_case(
                Default(test_group=self, resolution=resolution))
            self.add_test_case(
                RestartTest(test_group=self, resolution=resolution))
            self.add_test_case(
                ThreadsTest(test_group=self, resolution=resolution))


def configure(resolution, config):
    """
    Modify the configuration options for one of the baroclinic test cases

    Parameters
    ----------
    resolution : str
        The resolution of the test case

    config : compass.config.CompassConfigParser
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
        config.set('baroclinic_channel', param, '{}'.format(res_params[param]))
