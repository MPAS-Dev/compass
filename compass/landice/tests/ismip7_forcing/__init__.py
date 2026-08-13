from compass.landice.tests.ismip7_forcing.atmosphere import Atmosphere
from compass.landice.tests.ismip7_forcing.fracture import Fracture
from compass.landice.tests.ismip7_forcing.ocean_thermal import OceanThermal
from compass.testgroup import TestGroup


class Ismip7Forcing(TestGroup):
    """
    A test group for processing ISMIP7 forcing data
    for the Antarctic Ice Sheet
    """

    def __init__(self, mpas_core):
        """
        Create the test group

        Parameters
        ----------
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name="ismip7_forcing")

        self.add_test_case(Atmosphere(test_group=self))
        self.add_test_case(OceanThermal(test_group=self))
        self.add_test_case(Fracture(test_group=self))
