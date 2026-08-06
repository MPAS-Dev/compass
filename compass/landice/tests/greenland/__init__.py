from compass.landice.tests.greenland.decomposition_test import (
    DecompositionTest,
)
from compass.landice.tests.greenland.mesh_gen import MeshGen
from compass.landice.tests.greenland.restart_test import RestartTest
from compass.landice.tests.greenland.smoke_test import SmokeTest
from compass.testgroup import TestGroup


class Greenland(TestGroup):
    """
    A test group for Greenland test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='greenland')

        for velo_solver in ['sia', 'FO']:
            for advection_type in ['fo', 'fct']:
                self.add_test_case(
                    SmokeTest(test_group=self, velo_solver=velo_solver,
                              advection_type=advection_type))

                self.add_test_case(
                    DecompositionTest(test_group=self, velo_solver=velo_solver,
                                      advection_type=advection_type))

                self.add_test_case(
                    RestartTest(test_group=self, velo_solver=velo_solver,
                                advection_type=advection_type))

        self.add_test_case(
            MeshGen(test_group=self))
