from compass.testgroup import TestGroup
from compass.ocean.tests.overflow.default import Default
from compass.ocean.tests.overflow.rpe_test import RpeTest


class Overflow(TestGroup):
    """
    A test group for General Ocean Turbulence Model (GOTM) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='overflow')

        self.add_test_case(Default(test_group=self, resolution='10km'))
        self.add_test_case(RpeTest(test_group=self, resolution='2km'))


def configure(resolution, config):
    """
    Modify the configuration options for one of the overflow test cases

    Parameters
    ----------
    resolution : str
        The resolution of the test case

    config : compass.config.CompassConfigParser
        Configuration options for this test case
    """
    width = config.getint('overflow', 'width')
    length = config.getint('overflow', 'length')

    dc = float(resolution[:-2])
    nx = int(width/dc)
    ny = int(length/dc)

    config.set('overflow', 'nx', str(nx),
               comment='the number of mesh cells in the x direction')
    config.set('overflow', 'ny', str(ny),
               comment='the number of mesh cells in the y direction')
    config.set('overflow', 'dc', str(dc*1e3),
               comment='the distance between adjacent cell centers')
