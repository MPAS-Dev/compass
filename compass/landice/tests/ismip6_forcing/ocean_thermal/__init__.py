from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.ocean_thermal.\
    process_thermal_forcing import ProcessThermalForcing
from compass.landice.tests.ismip6_forcing.configure import configure as \
    configure_testgroup


class OceanThermal(TestCase):
    """
    A test case for processing the ISMIP6 ocean thermal forcing data.
    The test case builds a mapping file for interpolation between the
    ISMIP6 8km polarstereo grid and MALI mesh, regrids the forcing data
    and renames the ISMIP6 variables to corresponding MALI variables.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_forcing.Ismip6Forcing
            The test group that this test case belongs to
        """
        name = 'ocean_thermal'
        super().__init__(test_group=test_group, name=name)

        step = ProcessThermalForcing(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """
        process_obs_data = self.config.getboolean('ismip6_ais_ocean_thermal',
                                                  'process_obs_data')
        check_model_options = not process_obs_data
        configure_testgroup(config=self.config,
                            check_model_options=check_model_options)
