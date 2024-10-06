from compass.ocean.tests.utility.combine_topo import CombineTopo
from compass.ocean.tests.utility.cull_restarts import CullRestarts
from compass.ocean.tests.utility.extrap_woa import ExtrapWoa
from compass.testgroup import TestGroup


class Utility(TestGroup):
    """
    A test group for general ocean utilities
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='utility')

        for target_grid in ['lat_lon', 'cubed_sphere']:
            self.add_test_case(
                CombineTopo(test_group=self, target_grid=target_grid),
            )
        self.add_test_case(CullRestarts(test_group=self))
        self.add_test_case(ExtrapWoa(test_group=self))
