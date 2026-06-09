from compass.landice.tests.ismip7_run.ismip7_ais import Ismip7Ais
from compass.landice.tests.ismip7_run.ismip7_gris import Ismip7Gris
from compass.testgroup import TestGroup


class Ismip7Run(TestGroup):
    """
    A test group for automated setup of a suite of standardized
    ISMIP7 simulations for both AIS and GrIS.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip7_run')

        self.add_test_case(Ismip7Ais(test_group=self))
        self.add_test_case(Ismip7Gris(test_group=self))
