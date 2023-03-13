import os

from compass.landice.tests.ismip6_run.ismip6_ais_proj2300.set_up_experiment import (  # noqa
    SetUpExperiment,
)
from compass.testcase import TestCase


class Ismip6AisProj2300(TestCase):
    """
    A test case for automated setup of a suite of standardized
    simulations for ISMIP6-Projections2300-Antarctica
    See: https://www.climate-cryosphere.org/wiki/index.php?title=ISMIP6-Projections2300-Antarctica)  # noqa
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip6_run.Ismip6Run
            The test group that this test case belongs to

        """
        name = 'ismip6_ais_proj2300'

        super().__init__(test_group=test_group, name=name,
                         subdir=name)

    def configure(self):
        """
        Set up the desired ISMIP6 AIS 2300 experiments.

        Read the list from the config of which experiemnts the
        user wants to set up.  Call thee add_step method and add the
        experiment to steps_to_run.  Those operations are typically done
        in the constructor, but they are done here so that the list to set up
        can be adjusted in the config, and the config is not available until
        this point.
        """

        # user can specify any of: 'all', 'tier1', 'tier2', or a
        # comma-delimited list (or a single experiment)
        exp_list = self.config.get('ismip6_run_ais_2300', 'exp_list')
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

    # no run() method is needed

    # no validate() method is needed
