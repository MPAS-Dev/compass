from compass.landice.tests.ismip7_forcing.configure import (
    configure as configure_testgroup,
)
from compass.landice.tests.ismip7_forcing.ocean_thermal.build_3d_thermal_forcing import (  # noqa: E501
    BuildGreenland3dThermalForcing,
)
from compass.landice.tests.ismip7_forcing.ocean_thermal.process_thermal_forcing import (  # noqa: E501
    ProcessThermalForcing,
)
from compass.testcase import TestCase


class OceanThermal(TestCase):
    """
    A test case for processing ISMIP7 ocean thermal forcing data.
    For AIS: Remaps annual 3D thermal forcing from the ISMIP7 8km
    polar stereographic grid to the MALI mesh.
    For GrIS: Remaps monthly 2D thermal forcing from the ISMIP7 1km
    grid to the MALI mesh.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_forcing.Ismip7Forcing
            The test group that this test case belongs to
        """
        name = "ocean_thermal"
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(ProcessThermalForcing(test_case=self))
        self.add_step(BuildGreenland3dThermalForcing(test_case=self))

    def configure(self):
        """
        Configures test case
        """
        configure_testgroup(config=self.config)
