from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.mismipplus.run_model import RunModel


class SmokeTest(TestCase):
    """
    A test case for running a smoke test of the MISMIP+ configuration
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mismipplus.MISMIPplus
            The test group that this test case belongs to

        """
        name = 'smoke_test'
        super().__init__(test_group=test_group, name=name)

        ntasks = 36
        min_tasks = 4

        step = RunModel(test_case=self, name=name, subdir='simulation',
                        ntasks=ntasks, min_tasks=min_tasks, openmp_threads=1)
        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        # Comparing against itself to for a smoke test
        # (This allows the potential to compare against a baseline)
        variables = ['thickness', 'surfaceSpeed']
        compare_variables(test_case=self, variables=variables,
                          filename1='simulation/output.nc',
                          filename2='simulation/output.nc')
