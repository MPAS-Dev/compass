"""
Schedule restarts for incomplete ensemble runs.

This module provides utilities to create restart ensemble configurations
based on analysis results from a completed ensemble.
"""

import json
import os
from datetime import datetime


class RestartScheduler:
    """
    Create restart ensemble configuration based on analysis results.

    This class reads an analysis_summary.json from a completed ensemble,
    identifies runs needing restart, and generates configuration for a
    new restart_ensemble test case.
    """

    def __init__(self, summary_file, new_work_dir):
        """
        Initialize scheduler.

        Parameters
        ----------
        summary_file : str
            Path to analysis_summary.json from completed ensemble

        new_work_dir : str
            Directory where restart ensemble will be set up
        """
        self.summary_file = summary_file
        self.new_work_dir = new_work_dir

        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"Summary file not found: {summary_file}")

        with open(summary_file, 'r') as f:
            self.summary = json.load(f)

        self.original_ensemble_dir = self.summary['ensemble_dir']
        os.makedirs(new_work_dir, exist_ok=True)

    def identify_restart_candidates(
            self,
            min_years=50.0,
            max_attempts=3,
            verbose=True):
        """
        Identify runs that should be restarted.

        Parameters
        ----------
        min_years : float
            Minimum simulation years before restart (default: 50.0)
            Runs that haven't reached this threshold won't be restarted

        max_attempts : int
            Maximum restart attempts per run (default: 3)
            Prevents infinite restart loops

        verbose : bool
            Whether to print classification details

        Returns
        -------
        list
            Sorted list of run numbers to restart
        """
        restart_candidates = []

        for run_num in self.summary['restart_needed_runs']:
            results = self.summary['individual_results'].get(run_num, {})
            ss_info = results.get('steady_state', {})

            final_year = ss_info.get('metrics', {}).get('final_year', 0.0)

            if final_year >= min_years:
                # Check if run has too many restart attempts
                run_dir = os.path.join(
                    self.original_ensemble_dir, f'run{
                        run_num:03}')

                restart_attempts = 0
                if os.path.exists(run_dir):
                    restart_dirs = [d for d in os.listdir(run_dir)
                                    if d.startswith('restart_attempt_')]
                    restart_attempts = len(restart_dirs)

                if restart_attempts < max_attempts:
                    restart_candidates.append(run_num)
                    if verbose:
                        print(f"  run{run_num:03}: Restart candidate "
                              f"({final_year:.1f} yrs, \
                                      {restart_attempts} attempts)")
                else:
                    if verbose:
                        print(
                            f"run{run_num:03}: Max attempts reached \
                                    ({restart_attempts}/{max_attempts})")
            else:
                if verbose:
                    print(
                        f"  run{
                            run_num:03}: Too short ({
                            final_year:.1f} < {
                            min_years:.1f} yrs)")

        return sorted(restart_candidates)

    def create_config_file(self, restart_runs, base_config_file=None):
        """
        Create restart ensemble configuration file.

        Parameters
        ----------
        restart_runs : list
            Run numbers to restart

        base_config_file : str, optional
            Base configuration file to inherit settings from

        Returns
        -------
        str
            Path to created restart_ensemble.cfg
        """
        # Format the run list nicely
        run_list_str = ', '.join(map(str, restart_runs[:10]))
        if len(restart_runs) > 10:
            run_list_str += f', ... and {len(restart_runs) - 10} more'

        config_content = f"""# Restart ensemble configuration
# Created: {datetime.now().isoformat()}
#
# Original spinup ensemble: {self.original_ensemble_dir}
# Restarted from: {self.summary_file}
#
# Runs to restart ({len(restart_runs)} total):
# {run_list_str}

[ensemble_generator]
ensemble_template = sgh_ensemble

[restart_ensemble]

# Path to the spinup ensemble to restart from
spinup_work_dir = {self.original_ensemble_dir}

# Restart configuration
# Maximum consecutive restart attempts per run (prevents infinite loops)
max_consecutive_restarts = 3

# Minimum simulation length (years) before attempting restart
# Prevents restarting runs that are too short
min_simulation_years_before_restart = 50.0

# Whether to automatically restart incomplete runs
# Set to False for manual control
auto_restart_incomplete = True

# Analysis parameters (same as spinup_ensemble)
steady_state_window_years = 10.0
steady_state_imbalance_threshold = 0.05
balanced_accuracy_threshold = 0.65

# Specularity content TIFF file for validation (optional)
spec_tiff_file = None

[ensemble]

# Job parameters for restart jobs
ntasks = 128
cfl_fraction = 0.7
"""

        config_file = os.path.join(self.new_work_dir, 'restart_ensemble.cfg')
        with open(config_file, 'w') as f:
            f.write(config_content)

        print(f"Config file created: {config_file}")
        return config_file

    def print_summary(self, restart_runs):
        """
        Print restart scheduling summary.

        Parameters
        ----------
        restart_runs : list
            Run numbers identified for restart
        """
        print("\n" + "=" * 70)
        print("RESTART ENSEMBLE PLAN")
        print("=" * 70)
        print(f"Original ensemble: {self.original_ensemble_dir}")
        print(f"Restart work dir: {self.new_work_dir}")
        print()
        print(f"Runs to restart: {len(restart_runs)}")
        if restart_runs:
            # Print in groups of 10
            for i in range(0, len(restart_runs), 10):
                group = restart_runs[i:i + 10]
                print(f"  {group}")
        print()
        print(
            f"Already at steady state: \
                    {len(self.summary['steady_state_runs'])}")
        print(f"Data compatible: {len(self.summary['data_compatible_runs'])}")
        print(f"Both criteria met: {len(self.summary['both_criteria_runs'])}")
        print("=" * 70 + "\n")


def schedule_restarts(
        summary_file,
        new_work_dir,
        min_years=50.0,
        max_attempts=3):
    """
    Convenience function to schedule restarts from analysis summary.

    Parameters
    ----------
    summary_file : str
        Path to analysis_summary.json

    new_work_dir : str
        Directory where restart ensemble will be created

    min_years : float
        Minimum simulation years before restart

    max_attempts : int
        Maximum restart attempts per run

    Returns
    -------
    tuple
        (config_file, restart_runs) or (None, []) if no restarts needed

    Examples
    --------
    >>> from compass.landice.tests.ensemble_generator.
            ensemble_templates.sgh_ensemble.restart
            import schedule_restarts
    >>>
    >>> config_file, restart_runs = schedule_restarts(
    ...     '/work/ensemble1/spinup_ensemble/analysis_summary.json',
    ...     '/work/ensemble2',
    ...     min_years=50.0,
    ...     max_attempts=3
    ... )
    >>>
    >>> if config_file:
    ...     print(f"Restart config: {config_file}")
    ...     print(f"Runs to restart: {restart_runs}")
    """
    scheduler = RestartScheduler(summary_file, new_work_dir)

    print("Identifying restart candidates...")
    restart_runs = scheduler.identify_restart_candidates(
        min_years, max_attempts)

    if not restart_runs:
        print("No runs to restart!")
        return None, []

    scheduler.print_summary(restart_runs)

    config_file = scheduler.create_config_file(restart_runs)

    return config_file, restart_runs
