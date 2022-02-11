from compass.testgroup import TestGroup
from compass.landice.tests.humboldt.default import Default
from compass.landice.tests.humboldt.decomposition_test \
     import DecompositionTest
from compass.landice.tests.humboldt.restart_test import RestartTest


class Humboldt(TestGroup):
    """
    A test group for humboldt test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='humboldt')

        self.add_test_case(
            Default(test_group=self))

        mesh_type='default' # Not currently used but expected to be added

        for velo_solver in ['FO',]:

            self.add_test_case(
                    DecompositionTest(test_group=self,
                                      velo_solver=velo_solver,
                                      calving_law='none',
                                      mesh_type=mesh_type))

        # Create decomp tests for all calving laws.
        # Note that FO velo solver is NOT BFB across decompositions
        # currently, so instead test using 'none' (fixed velocity field from
        # input field) or 'sia'
        for velo_solver in ['none', 'FO',]:
            for calving_law in ['none', 'floating', 'eigencalving',
                                'specified_calving_velocity',
                                'von_Mises_stress', 'damagecalving', 
                                'ismip6_retreat']:

                self.add_test_case(
                    DecompositionTest(test_group=self,
                                      velo_solver=velo_solver,
                                      calving_law=calving_law,
                                      mesh_type=mesh_type))

                self.add_test_case(
                    RestartTest(test_group=self,
                                velo_solver=velo_solver,
                                calving_law=calving_law,
                                mesh_type=mesh_type))
