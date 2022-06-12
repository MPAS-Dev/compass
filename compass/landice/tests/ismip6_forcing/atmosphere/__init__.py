from compass.testcase import TestCase
from compass.landice.tests.ismip6_forcing.atmosphere.process_smb \
    import ProcessSMB
from compass.landice.tests.ismip6_forcing.atmosphere.process_smb_racmo \
    import ProcessSmbRacmo
from compass.landice.tests.ismip6_forcing.configure import configure as \
    configure_testgroup


class Atmosphere(TestCase):
    """
    A test case for processing the ISMIP6 atmosphere forcing data.
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
        name = 'atmosphere'
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        step = ProcessSMB(test_case=self)
        self.add_step(step)
        step = ProcessSmbRacmo(test_case=self)
        self.add_step(step)

    def configure(self):
        """
        Configures test case
        """

        configure_testgroup(config=self.config,
                            check_model_options=True)
