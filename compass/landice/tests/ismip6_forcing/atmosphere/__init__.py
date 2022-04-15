from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.atmosphere.process_smb \
    import ProcessSMB


class Atmosphere(TestCase):
    """
    A test case for processing the ISMIP6 atmosphere forcing data.
    The test case builds a mapping file for interpolation between the
    ISMIP6 8km polarstereo grid and MALI mesh, regrids the forcing data
    and rename the ISMIP6 variables to corresponding MALI variables.

    Attributes
    ----------
    mesh_type : str >>>>>>> what kind of attributes would we have?
        The resolution or type of mesh of the test case
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_forcing.Ismip6Forcing
            The test group that this test case belongs to
        """
        name = 'atmosphere'
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        step = ProcessSMB(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """
        input_path = self.config.get(section="ismip6_ais_atmosphere",
                                     option="input_path")
        if input_path == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais_atmosphere section"
                             "with the input_path option")
