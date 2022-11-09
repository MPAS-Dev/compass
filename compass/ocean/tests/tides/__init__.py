from compass.testgroup import TestGroup

from compass.ocean.tests.tides.mesh import Mesh
from compass.ocean.tests.tides.init import Init
from compass.ocean.tests.tides.forward import Forward


class Tides(TestGroup):
    """
    A test group for tidal simulations with MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='tides')

        for mesh_name in ['Icos7']:

            mesh = Mesh(test_group=self, mesh_name=mesh_name)
            self.add_test_case(mesh)

            init = Init(test_group=self, mesh=mesh)
            self.add_test_case(init)

            self.add_test_case(Forward(test_group=self,
                                       mesh=mesh,
                                       init=init))
