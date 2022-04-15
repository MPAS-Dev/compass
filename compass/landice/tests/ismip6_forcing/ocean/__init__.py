from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.ocean.process_basal_melt \
    import ProcessBasalMelt
from compass.landice.tests.ismip6_forcing.ocean.process_thermal_forcing \
    import ProcessThermalForcing


class Ocean(TestCase):
    """
    A test case for processing the ISMIP6 ocean forcing data.
    The test case builds a mapping file for interpolation between the
    ISMIP6 8km polarstereo grid and MALI mesh, regrids the forcing data
    and rename the ISMIP6 variables to corresponding MALI variables.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_forcing.Ismip6Forcing
            The test group that this test case belongs to
        """
        name = 'ocean'
        super().__init__(test_group=test_group, name=name)

        step = ProcessBasalMelt(test_case=self)
        self.add_step(step)

        step = ProcessThermalForcing(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """
        input_path = self.config.get(section="ismip6_ais_ocean",
                                     option="input_path")
        if input_path == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais_ocean section"
                             "with the input_path option")
