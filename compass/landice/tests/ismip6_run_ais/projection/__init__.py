from compass.testcase import TestCase
from compass.landice.tests.ismip6_run_ais.set_up_experiment \
        import SetUpExperiment
from compass.validate import compare_variables

import os


class Projection(TestCase):
    """
    A test case for performing forward MALI runs of ISMIP6 Antaractic setup
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_run_ais.Ismip6RunAIS
            The test group that this test case belongs to

        """
        name = 'ismip6AE'

        super().__init__(test_group=test_group, name=name,
                         subdir=name)

    def configure(self):
        """
        Set up the desired ISMIP6 experiments.

        Read the list from the config of which experiemnts the
        user wants to set up.  Call thee add_step method and add the
        experiment to steps_to_run.  Those operations are typically done
        in the constructor, but they are done here so that the list to set up
        can be adjusted in the config, and the config is not available until
        this point.
        """
        exp_list = self.config.get('ismip6_run_ais', 'exp_list')
        if exp_list == "tier1":
            exp_list = ['hist', 'ctrlAE'] + \
                       [f'expAE{i:02}' for i in range(1, 7)]
        elif exp_list == "tier2":
            exp_list = [f'expAE{i:02}' for i in range(7, 15)]
        elif exp_list == "all":
            exp_list = ['hist', 'ctrlAE'] + \
                       [f'expAE{i:02}' for i in range(1, 15)]
        else:
            exp_list = exp_list.split(",")

        for exp in exp_list:
            if os.path.exists(os.path.join(self.work_dir, exp)):
                print(f"WARNING: {exp} path already exists; skipping.  "
                      "Please remove the directory "
                      f"{os.path.join(self.work_dir, exp)} and execute "
                      "'compass setup' again to set this experiment up.")
            else:
                self.add_step(
                    SetUpExperiment(test_case=self, name=exp,
                                    subdir=exp, exp=exp))
                self.steps_to_run.append(exp)

    # no run() method is needed

    # no validate() method is needed
