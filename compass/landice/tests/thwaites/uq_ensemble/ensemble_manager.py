from compass.step import Step
from importlib.resources import path
from compass.io import symlink
import os
from mpas_tools.logging import check_call

class EnsembleManager(Step):
    """
    A step for an ensemble manager.
    The individual runs are other steps of this test case of class
    EnsembleMember

    Attributes
    ----------
    name : str
        the name of this step
    """

    def __init__(self, test_case):
        """
        Creates an ensemble manager

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        # define step name
        self.name = 'ensemble_manager'

        super().__init__(test_case=test_case, name=self.name)

    def setup(self):
        """
        Set up the ensemble manager
        """
        # Link viz script
        with path('compass.landice.tests.thwaites.uq_ensemble',
                  'plot_ensemble.py') as target:
            symlink(str(target), f'{self.test_case.work_dir}/plot_ensemble.py')

    def run(self):
        """
        Use the ensemble manager to manage and launch jobs for each run
        Each ensemble member is a step of the test case.
        The ensemble manager submits a job script for each run so that they
        are run in parallel through SLURM.
        Eventually we want this function to handle restarts.
        """
        logger = self.logger

        # Determine list of runs (steps) to loop over
        runs = list()
        for step in self.test_case.steps:
            if step != self.name:  # Ignore the step that is the run manager
                runs.append(step)

        # Now loop over runs and process each
        cwd = os.getcwd()  # save current working dir
        for run in runs:
            runStep = self.test_case.steps[run]  # Get step object from 'steps' dictionary
            os.chdir(runStep.work_dir)
            # TODO: assess if this run is unrun, partially run, or complete, and adjust accordingly
            check_call(['sbatch', 'job_script.sh'], logger)
            logger.info(f'Run {run} submitted.')
        logger.info(f'All runs submitted.')
