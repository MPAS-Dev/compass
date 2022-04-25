from compass.testgroup import TestGroup
from compass.landice.tests.ismip6_forcing.atmosphere import Atmosphere
from compass.landice.tests.ismip6_forcing.ocean import Ocean


class Ismip6Forcing(TestGroup):
    """
    A test group for processing the ISMIP6
         ocean and atmosphere forcing data
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='ismip6_forcing')

        self.add_test_case(Atmosphere(test_group=self))
        self.add_test_case(Ocean(test_group=self))
