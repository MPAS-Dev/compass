import os
import pickle
import sys

import numpy as np

from compass.landice.tests.ensemble_generator.branch_ensemble.branch_run import (  # noqa
    BranchRun,
)
from compass.landice.tests.ensemble_generator.ensemble_manager import (
    EnsembleManager,
)
from compass.testcase import TestCase


class BranchEnsemble(TestCase):
    """
    A test case for performing an ensemble of
    simulations for uncertainty quantification studies.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ensemble_generator.EnsembleGenerator
            The test group that this test case belongs to

        """
        name = 'branch_ensemble'
        super().__init__(test_group=test_group, name=name)

        # We don't want to initialize all the individual runs
        # So during init, we only add the run manager
        self.add_step(EnsembleManager(test_case=self))

    def configure(self):
        """
        Configure a parameter ensemble of MALI simulations.

        Start by identifying the start and end run numbers to set up
        from the config.

        Next, read a pre-defined unit parameter vector that can be used
        for assigning parameter values to each ensemble member.

        The main work is using the unit parameter vector to set parameter
        values for each parameter to be varied, over prescribed ranges.

        Then create the ensemble member as a step in the test case by calling
        the EnsembleMember constructor.

        Finally, add this step to the test case's step_to_run.  This normally
        happens automatically if steps are added to the test case in the test
        case constructor, but because we waited to add these steps until this
        configure phase, we must explicitly add the steps to steps_to_run.
        """

        config = self.config
        section = config['branch_ensemble']

        spinup_test_dir = section.get('spinup_test_dir')
        branch_year = section.getint('branch_year')

        # Determine start and end run numbers being requested
        self.start_run = section.getint('start_run')
        self.end_run = section.getint('end_run')

        # Determine whether to only set up filtered runs
        self.set_up_filtered_only = section.getboolean('set_up_filtered_only')
        self.ensemble_pickle_file = section.get('ensemble_pickle_file')
        if self.set_up_filtered_only:
            with open(self.ensemble_pickle_file, 'rb') as f:
                [param_info, qoi_info] = pickle.load(f)
            filtered_runs = np.isfinite(qoi_info['VAF change']['values'])
        else:
            filtered_runs = np.ones((self.end_run + 1,))

        for run_num in range(self.start_run, self.end_run + 1):
            run_name = f'run{run_num:03}'
            if (filtered_runs[run_num] and
                os.path.isfile(os.path.join(spinup_test_dir, run_name,
                                            f'rst.{branch_year}-01-01.nc'))):
                if os.path.exists(os.path.join(self.work_dir, run_name)):
                    print(f"WARNING: {run_name} path already exists; "
                          "skipping.  ")
                else:
                    print(f"Adding {run_name}")
                    # use this run
                    self.add_step(BranchRun(test_case=self, run_num=run_num))
                    # Note: do not add to steps_to_run; ensemble_manager
                    # will handle submitting and running the runs

        # Have compass run only run the run_manager but not any actual runs.
        # This is because the individual runs will be submitted as jobs
        # by the ensemble manager.
        self.steps_to_run = ['ensemble_manager',]

    # no run() method is needed

    # no validate() method is needed
