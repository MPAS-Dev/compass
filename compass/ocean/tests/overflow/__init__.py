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
        self.add_test_case(RpeTest(test_group=self, resolution='1km'))

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
    res_params = {'10km': {'nx': 4,
                           'ny': 24,
                           'dc': 10e3},
                  '1km': {'nx': 4,
                          'ny': 200,
                          'dc': 1e3}}

    comment = {'nx': 'the number of mesh cells in the x direction',
               'ny': 'the number of mesh cells in the y direction',
               'dc': 'the distance between adjacent cell centers'}

    if resolution not in res_params:
        raise ValueError(f'Unsupported resolution {resolution}. Supported '
                         f'values are: {list(res_params)}')

    res_params = res_params[resolution]
    for param in res_params:
        config.set('overflow', param, str(res_params[param]),
                   comment=comment[param])
