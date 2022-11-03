from compass.testgroup import TestGroup
from compass.ocean.tests.merry_go_round.default import Default

class MerryGoRound(TestGroup):
    """
    A test group for tracer advection test cases "merry-go-round"
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='merry_go_round')

        self.add_test_case(Default(test_group=self))

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
    res_params = {'5m': {'nx': 100,
                         'ny': 4,
                         'dc': 5}}

    comment = {'nx': 'the number of mesh cells in the x direction',
               'ny': 'the number of mesh cells in the y direction',
               'dc': 'the distance between adjacent cell centers'}

    if resolution not in res_params:
        raise ValueError(f'Unsupported resolution {resolution}. Supported '
                         f'values are: {list(res_params)}')
    res_params = res_params[resolution]
    for param in res_params:
        config.set('merry_go_round', param, str(res_params[param]),
                   comment=comment[param])
