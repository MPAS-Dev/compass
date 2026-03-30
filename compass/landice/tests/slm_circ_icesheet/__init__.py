from compass.landice.tests.slm_circ_icesheet.mesh_convergence import (
    MeshConvergenceTest,
)
from compass.landice.tests.slm_circ_icesheet.smoke_test import SmokeTest
from compass.testgroup import TestGroup


class SlmCircIcesheet(TestGroup):
    """
    This test group generates an idealized, circular ice sheet that has a
    prescribed thickness evolution for testing coupling between MALI
    and the Sea-Level Model.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='slm_circ_icesheet')

        self.add_test_case(
            MeshConvergenceTest(test_group=self))
        self.add_test_case(SmokeTest(test_group=self))
