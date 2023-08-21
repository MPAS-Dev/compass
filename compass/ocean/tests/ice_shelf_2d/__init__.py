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

        for resolution in [2e3, 5e3]:
            for coord_type in ['z-star', 'z-level', 'single_layer']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            coord_type=coord_type))
                self.add_test_case(
                    Default(test_group=self, resolution=resolution,
                            coord_type=coord_type, tidal_forcing=True))
                self.add_test_case(
                    RestartTest(test_group=self, resolution=resolution,
                                coord_type=coord_type))


def configure(resolution, coord_type, config):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    resolution : float
        The resolution of the test case in meters

    coord_type : str
        The type of vertical coordinate (``z-star``, ``z-level``, etc.)

    config : compass.config.CompassConfigParser
        Configuration options for this test case
    """
    dx = 50e3  # width of domain in m
    dy = 220e3  # length of domain in m
    dc = resolution
    nx = int(dx / resolution)
    # ny needs to be even because it is nonperiodic
    ny = 2 * int(dy / (2. * resolution))

    config.set('ice_shelf_2d', 'nx', f'{nx}')
    config.set('ice_shelf_2d', 'ny', f'{ny}')
    config.set('ice_shelf_2d', 'dc', f'{dc}')

    config.set('vertical_grid', 'coord_type', coord_type)
    if coord_type == 'z-level':
        # we need more vertical resolution
        config.set('vertical_grid', 'vert_levels', '100')
    elif coord_type == 'single_layer':
        config.set('vertical_grid', 'vert_levels', '1')
        config.set('vertical_grid', 'coord_type', 'z-level')
        config.set('vertical_grid', 'partial_cell_type', 'None')
