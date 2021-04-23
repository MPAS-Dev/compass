from compass.testcase import TestCase
from compass.landice.tests.greenland.run_model import RunModel


class SmokeTest(TestCase):
    """
    The default test case for the Greenland test group simply downloads the
    mesh and initial condition, then performs a short forward run on 4 cores.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.greenland.Greenland
            The test group that this test case belongs to
        """
        name = 'smoke_test'
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            RunModel(test_case=self, cores=4, threads=1))

    # no configure() method is needed because we will use the default dome
    # config options

    # no run() method is needed because we're doing the default: running all
    # steps
