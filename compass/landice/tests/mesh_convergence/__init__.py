from compass.landice.tests.mesh_convergence.halfar import Halfar
from compass.landice.tests.mesh_convergence.horizontal_advection import (
    HorizontalAdvection,
)
from compass.testgroup import TestGroup


class MeshConvergence(TestGroup):
    """
    A test group for convergence tests with MALI
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.LandIce
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='mesh_convergence')

        self.add_test_case(HorizontalAdvection(test_group=self))
        self.add_test_case(Halfar(test_group=self))
