from compass.landice.tests.dome.decomposition_test import DecompositionTest
from compass.landice.tests.dome.restart_test import RestartTest
from compass.landice.tests.dome.smoke_test import SmokeTest
from compass.testgroup import TestGroup


class Dome(TestGroup):
    """
    A test group for dome test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='dome')

        for mesh_type in ['2000m', 'variable_resolution']:
            for velo_solver in ['sia', 'FO']:
                for advection_type in ['fo', 'fct']:

                    self.add_test_case(
                        SmokeTest(test_group=self, velo_solver=velo_solver,
                                  mesh_type=mesh_type,
                                  advection_type=advection_type))

                    self.add_test_case(
                        DecompositionTest(test_group=self,
                                          velo_solver=velo_solver,
                                          mesh_type=mesh_type,
                                          advection_type=advection_type))

                    self.add_test_case(
                        RestartTest(test_group=self, velo_solver=velo_solver,
                                    mesh_type=mesh_type,
                                    advection_type=advection_type))
