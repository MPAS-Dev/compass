"""
Restart ensemble test case for SGH template.

This test case identifies incomplete runs from a spinup ensemble and sets up
restart steps for them. Each restart step continues the simulation from the
last checkpoint.

Usage:
    compass setup -t landice/ensemble_generator/sgh_restart_ensemble
        -w /work/restart -f restart_ensemble.cfg
    compass run -w /work/restart
"""

import glob
import json
import os

from compass.landice.tests.ensemble_generator.ensemble_manager import (
    EnsembleManager,
)
from compass.testcase import TestCase

from .restart_member import InPlaceRestartMember


class RestartEnsemble(TestCase):
    """
    A test case for restarting incomplete ensemble members.

    This identifies runs from a spinup_ensemble that did not complete
    or reach steady state, and continues them from their last checkpoint.
    """

    def __init__(self, test_group):
        """
        Create the restart ensemble test case

        Parameters
        ----------
        test_group : compass test group
            The test group that this test case belongs to
        """
        name = 'sgh_restart_ensemble'
        super().__init__(test_group=test_group, name=name)

        # Add the ensemble manager (handles job submission)
        self.add_step(EnsembleManager(test_case=self))

    def configure(self):
        """
        Configure restart ensemble by identifying incomplete runs.

        This method:
        1. Reads the spinup ensemble directory
        2. Checks analysis results to identify incomplete runs
        3. Creates RestartMember steps for runs needing continuation
        4. Sets up ensemble_manager to handle job submission
        """
        config = self.config
        section = config.get('restart_ensemble', {})

        spinup_work_dir = section.get('spinup_work_dir')

        if not spinup_work_dir:
            raise ValueError(
                "restart_ensemble config must specify spinup_work_dir\n"
                "Add to config file:\n"
                "[restart_ensemble]\n"
                "spinup_work_dir = /path/to/spinup/ensemble"
            )

        if not os.path.exists(spinup_work_dir):
            raise ValueError(f"spinup_work_dir not found: {spinup_work_dir}")

        # Get restart configuration
        max_consecutive_restarts = section.getint(
            'max_consecutive_restarts', 3)
        min_simulation_years = section.getfloat(
            'min_simulation_years_before_restart', 50.0)
        auto_restart = section.getboolean('auto_restart_incomplete', True)

        # Scan for existing run directories
        run_dirs = sorted(glob.glob(os.path.join(spinup_work_dir, 'run*')))

        restart_runs = []
        skipped_runs = []

        for run_dir in run_dirs:
            run_name = os.path.basename(run_dir)
            try:
                run_num = int(run_name.replace('run', ''))
            except ValueError:
                continue

            # Check if run should be restarted
            should_restart, reason = self._should_restart_run(
                run_dir=run_dir,
                run_num=run_num,
                min_years=min_simulation_years,
                max_restarts=max_consecutive_restarts,
                auto_restart=auto_restart
            )

            if should_restart:
                restart_runs.append(run_num)
                print(f"Scheduling restart for {run_name}")

                # Add restart member step
                self.add_step(InPlaceRestartMember(
                    test_case=self,
                    run_num=run_num,
                    spinup_work_dir=spinup_work_dir
                ))
            else:
                if reason:
                    skipped_runs.append((run_num, reason))

        if skipped_runs:
            print("\nSkipped runs:")
            for run_num, reason in skipped_runs:
                print(f"  run{run_num:03}: {reason}")

        self.restart_run_numbers = restart_runs

        # Only run ensemble_manager; it submits individual restart jobs
        self.steps_to_run = ['ensemble_manager']

    def _should_restart_run(
            self,
            run_dir,
            run_num,
            min_years,
            max_restarts,
            auto_restart):
        """
        Determine if a run should be restarted.

        Parameters
        ----------
        run_dir : str
            Directory of the original run

        run_num : int
            Run number

        min_years : float
            Minimum simulation years required before restart

        max_restarts : int
            Maximum number of restart attempts allowed

        auto_restart : bool
            Whether to automatically restart incomplete runs

        Returns
        -------
        tuple
            (should_restart, reason_if_skipped)
        """

        # Check if run has output
        output_file = os.path.join(run_dir, 'output', 'globalStats.nc')
        if not os.path.exists(output_file):
            return False, "No output file"

        # Check if run completed (reached stop time)
        restart_timestamp_file = os.path.join(run_dir, 'restart_timestamp')
        namelist_file = os.path.join(run_dir, 'namelist.landice')

        if not os.path.exists(restart_timestamp_file):
            return False, "No restart_timestamp (run may have failed)"

        try:
            with open(restart_timestamp_file, 'r') as f:
                current_time = f.read().strip()

            import compass.namelist
            namelist = compass.namelist.ingest(namelist_file)
            stop_time = \
                namelist['time_management']['config_stop_time'].strip(
                ).strip("'")

            if current_time == stop_time:
                return False, "Already completed"

        except Exception as e:
            return False, f"Error reading completion status: {e}"

        # Check analysis results
        analysis_file = os.path.join(run_dir, 'analysis_results.json')

        if os.path.exists(analysis_file):
            try:
                with open(analysis_file, 'r') as f:
                    results = json.load(f)

                # If at steady state, don't restart
                ss_info = results.get('steady_state', {})
                if ss_info.get('is_steady_state', False):
                    return False, "Already at steady state"

                # Check simulation length
                metrics = ss_info.get('metrics', {})
                sim_length = metrics.get('final_year', 0.0)

                if sim_length < min_years:
                    return False, f"Too short ({
                        sim_length:.1f} < {
                        min_years:.1f} yrs)"

            except (json.JSONDecodeError, IOError):
                # If analysis file is malformed, still allow restart
                pass
        else:
            # No analysis file - if we can't verify it reached min years, don't
            # restart
            return False, "No analysis results to verify progress"

        # Check number of restart attempts
        restart_attempts = 0
        if os.path.exists(run_dir):
            restart_dirs = [d for d in os.listdir(run_dir)
                            if d.startswith('restart_attempt_')]
            restart_attempts = len(restart_dirs)

        if restart_attempts >= max_restarts:
            return False, f"Max restart attempts reached \
                    ({restart_attempts}/{max_restarts})"

        # If all checks pass and auto_restart is enabled
        if not auto_restart:
            return False, "Auto-restart disabled"

        return True, None

    # no run() method is needed
    # no validate() method is needed
