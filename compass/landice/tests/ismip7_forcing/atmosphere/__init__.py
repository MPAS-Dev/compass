from compass.landice.tests.ismip7_forcing.atmosphere.process_smb import (
    ProcessSmb,
)
from compass.landice.tests.ismip7_forcing.atmosphere.process_smb_gradient import (  # noqa: E501
    ProcessSmbGradient,
)
from compass.landice.tests.ismip7_forcing.atmosphere.process_temperature import (  # noqa: E501
    ProcessTemperature,
)
from compass.landice.tests.ismip7_forcing.configure import (
    configure as configure_testgroup,
)
from compass.testcase import TestCase


class Atmosphere(TestCase):
    """
    A test case for processing ISMIP7 AIS atmosphere forcing data.
    Remaps monthly SMB, temperature, and annual SMB gradient from the
    ISMIP7 2km polar stereographic grid to the MALI unstructured mesh.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_forcing.Ismip7Forcing
            The test group that this test case belongs to
        """
        name = "atmosphere"
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(ProcessSmb(test_case=self))
        self.add_step(ProcessTemperature(test_case=self))
        self.add_step(ProcessSmbGradient(test_case=self))

    def configure(self):
        """
        Configures test case
        """
        configure_testgroup(config=self.config)
