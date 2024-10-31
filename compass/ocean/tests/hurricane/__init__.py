from compass.ocean.tests.hurricane.forward import Forward
from compass.ocean.tests.hurricane.init import Init
from compass.ocean.tests.hurricane.mesh import Mesh
from compass.testgroup import TestGroup


class Hurricane(TestGroup):
    """
    A test group for hurricane simulations with MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='hurricane')

        storm = 'sandy'
        mesh_names = ['DEQU120at30cr10rr2', 'DEVR45to5rr1']
        wetdry_options = ['off', 'standard', 'subgrid']

        for mesh_name in mesh_names:
            for wetdry in wetdry_options:
                for use_lts in [False, 'LTS', 'FB_LTS']:

                    mesh = Mesh(test_group=self,
                                mesh_name=mesh_name,
                                use_lts=use_lts)
                    self.add_test_case(mesh)

                    init = Init(test_group=self,
                                mesh=mesh,
                                storm=storm,
                                use_lts=use_lts,
                                wetdry=wetdry)
                    self.add_test_case(init)

                    forward = Forward(test_group=self,
                                      mesh=mesh,
                                      storm=storm,
                                      init=init,
                                      use_lts=use_lts,
                                      wetdry=wetdry)
                    self.add_test_case(forward)
