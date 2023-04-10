from compass.landice.tests.ismip6_run.ismip6_ais_proj2300 import (
    Ismip6AisProj2300,
)
from compass.testgroup import TestGroup


class Ismip6Run(TestGroup):
    """
    A test group for automated setup of a suite of standardized
    ISMIP6 simulations

    Attributes
    ----------
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip6_run')

        self.add_test_case(Ismip6AisProj2300(test_group=self))
