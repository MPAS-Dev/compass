from compass.testgroup import TestGroup
from compass.landice.tests.dome.smoke_test import SmokeTest
from compass.landice.tests.dome.decomposition_test import DecompositionTest
from compass.landice.tests.dome.restart_test import RestartTest


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
            self.add_test_case(
                SmokeTest(test_group=self, velo_solver=velo_solver, mesh_type=mesh_type))
            self.add_test_case(
                DecompositionTest(test_group=self, velo_solver=velo_solver, mesh_type=mesh_type))
            self.add_test_case(
                RestartTest(test_group=self, velo_solver=velo_solver, mesh_type=mesh_type))
