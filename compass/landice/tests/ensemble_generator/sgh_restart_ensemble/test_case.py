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

import compass.namelist
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
        # Load shipped defaults before reading any options so that the
        # compass config infrastructure (MpasConfigParser) can resolve them
        # without needing fallback= kwargs.
        self.config.add_from_package(
            'compass.landice.tests.ensemble_generator.sgh_restart_ensemble',
            'ensemble_generator.cfg')
        # Bug fix: use dict-style access to get a SectionProxy, not .get()
        section = config['restart_ensemble']

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
        max_consecutive_restarts = section.getint('max_consecutive_restarts')
        min_simulation_years = section.getfloat(
            'min_simulation_years_before_restart')
        auto_restart = section.getboolean('auto_restart_incomplete')

        # Load per-run analysis results from analysis_summary.json if provided.
        # The sgh_ensemble_analysis test case writes this file; it contains
        # an 'individual_results' dict keyed by run number (strings in JSON).
        analysis_summary = {}
        analysis_summary_file = section.get('analysis_summary_file')
        if analysis_summary_file and \
                analysis_summary_file.lower() != 'none':
            if not os.path.exists(analysis_summary_file):
                raise ValueError(
                    "analysis_summary_file not found: "
                    f"{analysis_summary_file}")
            with open(analysis_summary_file, 'r') as f:
                summary = json.load(f)
            # individual_results keys are ints in Python but strings in JSON
            analysis_summary = summary.get('individual_results', {})

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

            # Look up per-run results from the summary (JSON int keys → str)
            run_results = analysis_summary.get(str(run_num))

            # Check if run should be restarted
            should_restart, reason = self._should_restart_run(
                run_dir=run_dir,
                run_num=run_num,
                min_years=min_simulation_years,
                max_restarts=max_consecutive_restarts,
                auto_restart=auto_restart,
                run_results=run_results
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
            auto_restart,
            run_results=None):
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

        run_results : dict or None
            Per-run results dict from ``individual_results[run_num]`` in
            ``analysis_summary.json``.  When *None* the run cannot be
            verified and will be skipped.

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

            namelist = compass.namelist.ingest(namelist_file)
            stop_time = \
                namelist['time_management']['config_stop_time'].strip(
                ).strip("'")

            if current_time == stop_time:
                return False, "Already completed"

        except Exception as e:
            return False, f"Error reading completion status: {e}"

        # Check analysis results supplied from analysis_summary.json.
        # run_results is the per-run dict from individual_results[run_num].
        if run_results is not None:
            ss_info = run_results.get('steady_state') or {}

            # If at steady state, don't restart
            if ss_info.get('is_steady_state', False):
                return False, "Already at steady state"

            # Check simulation length
            metrics = ss_info.get('metrics') or {}
            sim_length = metrics.get('final_year', 0.0)

            if sim_length < min_years:
                return False, (f"Too short "
                               f"({sim_length:.1f} < {min_years:.1f} yrs)")
        else:
            # No analysis results — cannot verify progress; skip
            return False, "No analysis results to verify progress"

        # Check number of restart attempts (tracked as restart_attempt_N/ dirs)
        restart_attempts = 0
        if os.path.exists(run_dir):
            restart_dirs = [d for d in os.listdir(run_dir)
                            if d.startswith('restart_attempt_')]
            restart_attempts = len(restart_dirs)

        if restart_attempts >= max_restarts:
            return False, (f"Max restart attempts reached "
                           f"({restart_attempts}/{max_restarts})")

        # If all checks pass and auto_restart is enabled
        if not auto_restart:
            return False, "Auto-restart disabled"

        return True, None

    # no run() method is needed
    # no validate() method is needed
