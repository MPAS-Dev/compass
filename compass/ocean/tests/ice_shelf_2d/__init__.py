from compass.ocean.tests.ice_shelf_2d.default import Default
from compass.ocean.tests.ice_shelf_2d.restart_test import RestartTest
from compass.testgroup import TestGroup


class IceShelf2d(TestGroup):
    """
    A test group for ice-shelf 2D test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ice_shelf_2d')

        for resolution in ['5km']:
            for coord_type in ['z-star', 'z-level']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            coord_type=coord_type))
                self.add_test_case(
                    RestartTest(test_group=self, resolution=resolution,
                                coord_type=coord_type))


def configure(resolution, coord_type, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    resolution : str
        The resolution of the test case

    coord_type : str
        The type of vertical coordinate (``z-star``, ``z-level``, etc.)

    config : compass.config.CompassConfigParser
        Configuration options for this test case
    """
    res_params = {'5km': {'nx': 10, 'ny': 44, 'dc': 5e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('ice_shelf_2d', param, '{}'.format(res_params[param]))

    config.set('vertical_grid', 'coord_type', coord_type)
    if coord_type == 'z-level':
        # we need more vertical resolution
        config.set('vertical_grid', 'vert_levels', '100')
