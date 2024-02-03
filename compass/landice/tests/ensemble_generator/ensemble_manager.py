import glob
import os
from importlib.resources import path

from mpas_tools.logging import check_call

import compass.namelist
from compass.io import symlink
from compass.step import Step


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
        Sets up the ensemble manager
        """
        # Link viz script
        with path('compass.landice.tests.ensemble_generator',
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
        for run in runs:
            # Get step object from 'steps' dictionary
            runStep = self.test_case.steps[run]
            os.chdir(runStep.work_dir)

            # Skip run if an error is detected
            err_files = glob.glob('log.landice.*.err')
            if len(err_files) > 0:
                print(f'{run} created an error log file in the '
                      'previous job and will not be restarted.')
                continue
            # Skip run if completed
            if os.path.isfile('restart_timestamp'):
                with open('restart_timestamp') as f:
                    curr_time = f.readline().strip()
                namelist = compass.namelist.ingest('namelist.landice')
                stop_time = (namelist['time_management']
                             ['config_stop_time'].strip().strip("'"))
                if curr_time == stop_time:
                    print(f'{run} has reached stop time.  Skipping.')
                    continue
            # If we didn't skip this run, submit it
            check_call(['sbatch', 'job_script.sh'], logger)
            logger.info(f'{run} submitted.')
        logger.info('All runs submitted.')
