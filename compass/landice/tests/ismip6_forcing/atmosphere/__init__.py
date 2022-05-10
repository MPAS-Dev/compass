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
        base_path_ismip6 = self.config.get(section="ismip6_ais",
                                           option="base_path_ismip6")
        base_path_mali = self.config.get(section="ismip6_ais",
                                         option="base_path_mali")
        mali_mesh_name = self.config.get(section="ismip6_ais",
                                         option="mali_mesh_name")
        mali_mesh_file = self.config.get(section="ismip6_ais",
                                         option="mali_mesh_file")

        if base_path_ismip6 == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the base_path_ismip6 option")
        if base_path_mali == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the base_path_mali option")
        if mali_mesh_name == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the mali_mesh_name option")
        if mali_mesh_file == "NotAvailable":
            raise ValueError("You need to supply a user config file, which "
                             "should contain the ismip6_ais "
                             "section with the mali_mesh_file option")
