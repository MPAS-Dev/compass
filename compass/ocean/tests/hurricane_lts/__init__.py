from compass.ocean.tests.hurricane_lts.forward import Forward
from compass.ocean.tests.hurricane_lts.init import Init
from compass.ocean.tests.hurricane_lts.mesh import Mesh
from compass.testgroup import TestGroup


class Hurricane_LTS(TestGroup):
    """
    A test group for hurricane simulations using
    local time-stepping (LTS) with MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='hurricane_lts')

        storm = 'sandy'
        mesh_name = 'DEQU120at30cr10rr2'
        mesh = Mesh(test_group=self, mesh_name=mesh_name)
        self.add_test_case(mesh)

        init = Init(test_group=self, mesh=mesh, storm=storm)
        self.add_test_case(init)

        self.add_test_case(Forward(test_group=self,
                                   mesh=mesh,
                                   storm=storm,
                                   init=init))
