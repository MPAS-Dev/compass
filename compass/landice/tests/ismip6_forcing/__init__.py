from compass.testgroup import TestGroup
from compass.landice.tests.ismip6_forcing.atmosphere import Atmosphere
from compass.landice.tests.ismip6_forcing.ocean_thermal import OceanThermal
from compass.landice.tests.ismip6_forcing.ocean_basal import OceanBasal
from compass.landice.tests.ismip6_forcing.shelf_collapse import ShelfCollapse

class Ismip6Forcing(TestGroup):
    """
    A test group for processing the ISMIP6 atmosphere, ocean and
    shelf-collapse forcing data
    """

    def __init__(self, mpas_core):
        """
        Create the test group

        Parameters
        ----------
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name="ismip6_forcing")

        self.add_test_case(Atmosphere(test_group=self))
        self.add_test_case(OceanBasal(test_group=self))
        self.add_test_case(OceanThermal(test_group=self, process_obs=True))
        self.add_test_case(OceanThermal(test_group=self, process_obs=False))
        self.add_test_case(ShelfCollapse(test_group=self))
