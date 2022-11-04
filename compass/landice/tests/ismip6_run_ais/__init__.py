from compass.testgroup import TestGroup
from compass.landice.tests.ismip6_run_ais.projection import Projection


class Ismip6RunAIS(TestGroup):
    """
    A test group for Antarctica forward simulation testcases

    Attributes
    ----------
    meshdirs : dict
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip6_run_ais')

        self.add_test_case(Projection(test_group=self))
