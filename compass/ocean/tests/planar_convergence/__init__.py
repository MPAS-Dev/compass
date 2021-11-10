from compass.testgroup import TestGroup
from compass.ocean.tests.planar_convergence.horizontal_advection import \
    HorizontalAdvection


class PlanarConvergence(TestGroup):
    """
    A test group for testing horizontal advection in MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='planar_convergence')

        self.add_test_case(HorizontalAdvection(test_group=self))
