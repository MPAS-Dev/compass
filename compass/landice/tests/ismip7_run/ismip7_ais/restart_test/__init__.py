from compass.landice.tests.ismip7_run.ismip7_ais.run_model import RunModel
from compass.parallel import get_available_parallel_resources
from compass.testcase import TestCase
from compass.validate import compare_variables


class RestartTest(TestCase):
    """
    A test case for performing two short MALI runs of the ISMIP7 AIS
    configuration: one full run and one run broken into two segments
    with a restart in between. The test case verifies that the results
    of the two runs are identical.

    Attributes
    ----------
    target_ntasks : int
        The preferred task count for the runs before resource
        constraints
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_run.Ismip7Run
            The test group that this test case belongs to
        """
        name = 'restart_test'
        super().__init__(test_group=test_group, name=name, subdir=name)
        self.target_ntasks = 128

        name = 'full_run'
        self.add_step(
            RunModel(test_case=self, name=name, subdir=name,
                     ntasks=self.target_ntasks, min_tasks=10))

        name = 'restart_run'
        self.add_step(
            RunModel(test_case=self, name=name, subdir=name,
                     ntasks=self.target_ntasks, min_tasks=10,
                     suffixes=['landice', 'landice.rst']))

    def configure(self):
        """
        Set restart-test task counts from framework-detected resources.

        The target task count is 128 when available. The FO velocity
        solver used by this configuration requires at least 10 tasks.
        """
        available_resources = get_available_parallel_resources(self.config)

        min_tasks = 10
        ntasks = max(min_tasks,
                     min(self.target_ntasks, available_resources['cores']))

        # Apply the same task count to both the full and restart runs.
        for step in self.steps.values():
            step.set_resources(ntasks=ntasks, min_tasks=min_tasks)

    # no run() method is needed

    def validate(self):
        """
        Compare the results of the full run and the restart run
        """
        variables = ['thickness', 'surfaceSpeed', 'calvingVelocity',
                     'calvingThickness', 'floatingBasalMassBal']

        compare_variables(
            test_case=self, variables=variables,
            filename1='full_run/output/output_2d_2000.nc',
            filename2='restart_run/output/output_2d_2000.nc')
