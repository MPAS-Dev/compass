from compass.landice.tests.ismip6_forcing.configure import (
    configure as configure_testgroup,
)
from compass.landice.tests.ismip6_forcing.ocean_thermal.process_thermal_forcing import (  # noqa: E501
    ProcessThermalForcing,
)
from compass.testcase import TestCase


class OceanThermal(TestCase):
    """
    A test case for processing the ISMIP6 ocean thermal forcing data.
    The test case builds a mapping file for interpolation between the
    ISMIP6 8km polarstereo grid and MALI mesh, regrids the forcing data
    and renames the ISMIP6 variables to corresponding MALI variables.

    Attributes
    ----------
    process_obs : bool
        Whether we are processing observations rather than CMIP model data
    """

    def __init__(self, test_group, process_obs):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_forcing.Ismip6Forcing
            The test group that this test case belongs to

        process_obs: bool
            Whether we are processing observations rather than CMIP model data
        """
        if process_obs:
            name = "ocean_thermal_obs"
        else:
            name = "ocean_thermal"
        self.process_obs = process_obs
        super().__init__(test_group=test_group, name=name)

        step = ProcessThermalForcing(test_case=self,
                                     process_obs=self.process_obs)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """
        check_model_options = not self.process_obs
        configure_testgroup(config=self.config,
                            check_model_options=check_model_options)
