import os

from compass.landice.tests.ismip7_run.ismip7_gris.set_up_experiment import (
    SetUpExperiment,
)
from compass.testcase import TestCase

# Define the full experiment matrix for GrIS
EXPERIMENTS = {
    'historical_CESM2-WACCM': {
        'scenario': 'historical', 'model': 'CESM2-WACCM',
        'start_time': '2000-01-15_00:00:00',
        'stop_time': '2015-01-01_00:00:00',
        'is_historical': True},
    'historical_MRI-ESM2-0': {
        'scenario': 'historical', 'model': 'MRI-ESM2-0',
        'start_time': '2000-01-15_00:00:00',
        'stop_time': '2015-01-01_00:00:00',
        'is_historical': True},
    'ssp370_CESM2-WACCM': {
        'scenario': 'ssp370', 'model': 'CESM2-WACCM',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2101-01-01_00:00:00',
        'is_historical': False},
    'ssp370_MRI-ESM2-0': {
        'scenario': 'ssp370', 'model': 'MRI-ESM2-0',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2101-01-01_00:00:00',
        'is_historical': False},
    'ssp126_CESM2-WACCM': {
        'scenario': 'ssp126', 'model': 'CESM2-WACCM',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ssp126_MRI-ESM2-0': {
        'scenario': 'ssp126', 'model': 'MRI-ESM2-0',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ssp585_CESM2-WACCM': {
        'scenario': 'ssp585', 'model': 'CESM2-WACCM',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ssp585_MRI-ESM2-0': {
        'scenario': 'ssp585', 'model': 'MRI-ESM2-0',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ctrl_CESM2-WACCM': {
        'scenario': 'ctrl', 'model': 'CESM2-WACCM',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ctrl_MRI-ESM2-0': {
        'scenario': 'ctrl', 'model': 'MRI-ESM2-0',
        'start_time': '2015-01-15_00:00:00',
        'stop_time': '2301-01-01_00:00:00',
        'is_historical': False},
    'ocx': {
        'scenario': 'ocx', 'model': None,
        'start_time': '1990-01-15_00:00:00',
        'stop_time': '2026-01-01_00:00:00',
        'is_historical': True},
}


class Ismip7Gris(TestCase):
    """
    A test case for automated setup of a suite of standardized
    ISMIP7 simulations for the Greenland Ice Sheet.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_run.Ismip7Run
            The test group that this test case belongs to
        """
        name = 'ismip7_gris'
        super().__init__(test_group=test_group, name=name, subdir=name)

    def configure(self):
        """
        Set up the desired ISMIP7 GrIS experiments.
        """
        config = self.config
        exp_list_str = config.get('ismip7_run_gris', 'exp_list')

        if exp_list_str == 'all':
            exp_list = list(EXPERIMENTS.keys())
        elif exp_list_str == 'historical':
            exp_list = [k for k, v in EXPERIMENTS.items()
                        if v['is_historical']]
        elif exp_list_str == 'projections':
            exp_list = [k for k, v in EXPERIMENTS.items()
                        if not v['is_historical'] and
                        v['scenario'] != 'ctrl']
        elif exp_list_str == 'ctrl':
            exp_list = [k for k, v in EXPERIMENTS.items()
                        if v['scenario'] == 'ctrl']
        else:
            exp_list = [s.strip() for s in exp_list_str.split(',')]

        for exp in exp_list:
            if exp not in EXPERIMENTS:
                raise ValueError(
                    f"Unknown experiment '{exp}'. Valid experiments: "
                    f"{list(EXPERIMENTS.keys())}")
            if os.path.exists(os.path.join(self.work_dir, exp)):
                print(f"WARNING: {exp} path already exists; skipping. "
                      "Remove the directory "
                      f"{os.path.join(self.work_dir, exp)} and run "
                      "'compass setup' again to recreate.")
            else:
                self.add_step(
                    SetUpExperiment(test_case=self, name=exp,
                                    subdir=exp, exp=exp,
                                    exp_info=EXPERIMENTS[exp]))

        # Do not add experiments to steps_to_run
        self.steps_to_run = []

    def run(self):
        """
        A dummy run method
        """
        raise ValueError(
            "ERROR: 'compass run' has no functionality at the test case "
            "level for this test. Please submit the job script in each "
            "experiment's subdirectory manually instead.")
