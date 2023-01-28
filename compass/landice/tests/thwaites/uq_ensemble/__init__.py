from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.thwaites.uq_ensemble.ensemble_member \
        import EnsembleMember
from compass.landice.tests.thwaites.uq_ensemble.ensemble_manager \
        import EnsembleManager


class UQEnsemble(TestCase):
    """
    A test case for performing two MALI runs of the Thwaites setup,
    with two different core counts.  The test case verifies that the
    results of the two runs are identical.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.thwaites.Thwaites
            The test group that this test case belongs to

        """
        name = 'thwaites_uq'
        super().__init__(test_group=test_group, name=name)

        # We don't want to initialize all the individual runs
        # So during init, we only add the run manager
        self.add_step(EnsembleManager(test_case=self))


    def configure(self):

        # add runs as steps based on the run range requested
        self.start_run = self.config.getint('thwaites_uq', 'start_run')
        self.end_run = self.config.getint('thwaites_uq', 'end_run')
        for run_num in range(self.start_run, self.end_run+1):
            self.add_step(EnsembleMember(test_case=self, run_num=run_num))
            # Note: do not add to step_to_run, because ensemble_manager
            # will handle submitting and running the runs

        # Have compass run only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]


    # no run() method is needed

    # no validate() method is needed
