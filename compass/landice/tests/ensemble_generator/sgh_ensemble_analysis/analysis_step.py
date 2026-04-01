"""
Analysis step that performs the actual ensemble analysis.
"""

import glob
import json
import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np

from compass.step import Step


def _sanitize_for_json(obj):
    """Recursively convert numpy types to native
    Python types for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class AnalysisStep(Step):
    """
    A step that analyzes a completed ensemble.
    """

    def __init__(self, test_case, ensemble_dir):
        """
        Create an analysis step.

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        ensemble_dir : str
            Directory containing completed ensemble runs
        """
        self.ensemble_dir = ensemble_dir

        super().__init__(test_case=test_case, name='analyze_ensemble')

    def setup(self):
        """Setup phase - prepare for analysis."""
        # Get path to analysis scripts in this package
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def run(self):
        """Run the analysis."""
        logger = self.logger

        logger.info(f"Analyzing ensemble: {self.ensemble_dir}")

        config = self.config

        # Read steady_state config, using has_option to handle missing keys
        ss_section = 'steady_state'
        window_years = (config.getfloat(ss_section, 'window_years')
                        if config.has_option(ss_section, 'window_years')
                        else 10.0)
        imbalance_threshold = (
            config.getfloat(ss_section, 'imbalance_threshold')
            if config.has_option(ss_section, 'imbalance_threshold')
            else 0.05)
        plot_results = (config.getboolean(ss_section, 'plot_results')
                        if config.has_option(ss_section, 'plot_results')
                        else False)

        # Read validation config
        val_section = 'validation'
        ba_threshold = (
            config.getfloat(val_section, 'balanced_accuracy_threshold')
            if config.has_option(val_section, 'balanced_accuracy_threshold')
            else 0.65)
        spec_tiff_file = (
            config.get(val_section, 'spec_tiff_file')
            if config.has_option(val_section, 'spec_tiff_file')
            else None)
        # treat the string 'None' as actual None
        if spec_tiff_file is not None and spec_tiff_file.lower() == 'none':
            spec_tiff_file = None
        plot_validation = (
            config.getboolean(val_section, 'plot_validation')
            if config.has_option(val_section, 'plot_validation')
            else False)

        analysis_config = {
            'steady_state': {
                'window_years': window_years,
                'imbalance_threshold': imbalance_threshold,
                'plot_results': plot_results,
            },
            'validation': {
                'balanced_accuracy_threshold': ba_threshold,
                'spec_tiff_file': spec_tiff_file,
                'plot_validation': plot_validation,
            },
        }

        logger.info(f"Loaded steady_state config: "
                    f"{analysis_config['steady_state']}")
        logger.info(f"Loaded validation config: "
                    f"{analysis_config['validation']}")

        # Create top-level figures directory next to analysis_summary.json
        figures_dir = os.path.join(self.work_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        # Initialize results
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_dir': self.ensemble_dir,
            'total_runs': 0,
            'completed_runs': 0,
            'incomplete_runs': 0,
            'steady_state_runs': [],
            'not_steady_state_runs': [],
            'data_compatible_runs': [],
            'not_data_compatible_runs': [],
            'both_criteria_runs': [],
            'restart_needed_runs': [],
            'individual_results': {},
            'analysis_parameters': {
                'steady_state': analysis_config.get(
                    'steady_state', {}),
                'validation': analysis_config.get(
                    'validation', {}),
            }
        }

        # Get all runs with output
        all_runs = self._get_all_runs()
        summary['total_runs'] = len(all_runs)

        logger.info(f"Found {len(all_runs)} total runs")
        logger.info("Checking for output files...")

        runs_with_output = []
        runs_without_output = []

        for run_dir in all_runs:
            run_name = os.path.basename(run_dir)
            run_num = int(run_name.replace('run', ''))

            if self._is_run_complete(run_dir):
                runs_with_output.append((run_num, run_dir, run_name))
            else:
                runs_without_output.append(run_num)

        summary['completed_runs'] = len(runs_with_output)
        summary['incomplete_runs'] = len(runs_without_output)

        logger.info(f"  {len(runs_with_output)} with output, "
                    f"{len(runs_without_output)} without output")
        logger.info("Analyzing runs with output...")

        # Analyze each run with output
        for run_num, run_dir, run_name in runs_with_output:
            # Create per-run figure subdirectory
            run_fig_dir = os.path.join(figures_dir, run_name)
            os.makedirs(run_fig_dir, exist_ok=True)

            results = self._run_analysis_on_run(
                run_dir, run_name, analysis_config, run_fig_dir)
            summary['individual_results'][run_num] = results

            # Categorize based on steady state
            ss_info = results.get('steady_state')
            val_info = results.get('validation')

            if ss_info and ss_info.get('is_steady_state'):
                summary['steady_state_runs'].append(run_num)
                # If steady state, also check validation
                if val_info and val_info.get('is_data_compatible'):
                    summary['data_compatible_runs'].append(run_num)
                    summary['both_criteria_runs'].append(run_num)
                elif (val_info and 'error' not in val_info and
                      val_info.get('status') != 'no_spec_file'):
                    summary['not_data_compatible_runs'].append(run_num)
            else:
                # Not steady state - needs restart
                summary['not_steady_state_runs'].append(run_num)
                summary['restart_needed_runs'].append(run_num)

        # Print summary
        self._print_summary(summary, logger)

        # Save summary to work directory
        summary_file = os.path.join(self.work_dir,
                                    'analysis_summary.json')
        summary = _sanitize_for_json(summary)
        with open(summary_file, 'w') as f:
            try:
                json.dump(summary, f, indent=2)
            except TypeError as err:
                raise ValueError(
                    f"Results contain unserializable values: {err}")
        logger.info(f"Summary saved to {summary_file}")

    def _get_all_runs(self):
        """Get sorted list of run directories."""
        run_dirs = sorted(glob.glob(
            os.path.join(self.ensemble_dir, 'run*')))
        return [d for d in run_dirs if os.path.isdir(d)]

    def _is_run_complete(self, run_dir):
        """Check if a run has completed successfully."""
        output_file = os.path.join(run_dir, 'output',
                                   'globalStats.nc')

        return os.path.exists(output_file)

    def _find_latest_output_file(self, output_dir):
        """Find the most recently modified output*.nc file in output_dir."""
        files = glob.glob(os.path.join(output_dir, 'output*.nc'))
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def _run_analysis_on_run(self, run_dir, run_name,
                             analysis_config, run_fig_dir):
        """Run analysis on a completed run."""
        self.logger.info(f"  Analyzing {run_name}...")

        output_file = os.path.join(run_dir, 'output',
                                   'globalStats.nc')
        results = {
            'run_name': run_name,
            'output_exists': os.path.exists(output_file),
            'analysis_timestamp': datetime.now().isoformat(),
            'steady_state': None,
            'validation': None,
            'analysis_errors': []
        }

        # Steady state analysis
        try:
            ss_config = analysis_config.get('steady_state', {})
            window_years = ss_config.get('window_years', 10.0)
            imbalance_threshold = ss_config.get(
                'imbalance_threshold', 0.05)
            plot_results = ss_config.get('plot_results', False)

            ss_results = self._run_steadystate_analysis(
                output_file, window_years, imbalance_threshold,
                plot_results, run_fig_dir)
            results['steady_state'] = ss_results

        except Exception as e:
            results['analysis_errors'].append(
                f"Steady-state analysis failed: {e}")
            self.logger.warning(f"    {e}")

        # Validation analysis
        try:
            val_config = analysis_config.get('validation', {})
            spec_tiff = val_config.get('spec_tiff_file', None)
            ba_threshold = val_config.get(
                'balanced_accuracy_threshold', 0.65)
            plot_validation = val_config.get(
                'plot_validation', False)

            if spec_tiff and os.path.exists(spec_tiff):
                output_dir = os.path.join(run_dir, 'output')
                latest_output = self._find_latest_output_file(output_dir)
                if latest_output is None:
                    raise ValueError(
                        f"No output*.nc files found in {output_dir}")
                val_results = self._run_validation_analysis(
                    latest_output, spec_tiff, ba_threshold,
                    plot_validation, run_fig_dir)
                results['validation'] = val_results
            else:
                results['validation'] = {'status': 'no_spec_file',
                                         'spec_tiff': spec_tiff}

        except Exception as e:
            results['analysis_errors'].append(
                f"Validation analysis failed: {e}")
            self.logger.warning(f"    {e}")

        return results

    def _run_steadystate_analysis(self, output_file, window_years,
                                  imbalance_threshold, plot=False,
                                  plot_dir=None):
        """Run steady-state analysis via subprocess."""
        script = os.path.join(
            self.script_dir,
            'analyze_subglacial_water_mass_balance.py')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            temp_json = f.name

        try:
            cmd = [
                'python', script,
                '-f', output_file,
                '--window_years', str(window_years),
                '--imbalance_threshold', str(imbalance_threshold),
                '--output_json', temp_json,
            ]

            if plot:
                cmd.append('--plot')
                if plot_dir is not None:
                    cmd += ['--plot_dir', plot_dir]

            result = subprocess.run(cmd, capture_output=True,
                                    text=True)

            if result.returncode == 0 and os.path.exists(temp_json):
                with open(temp_json, 'r') as f:
                    return json.load(f)
            else:
                raise RuntimeError(
                    f"Subprocess analysis failed: "
                    f"{result.stderr}")

        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)

    def _run_validation_analysis(self, output_file, spec_tiff,
                                 ba_threshold, plot=False,
                                 plot_dir=None):
        """Run validation analysis via subprocess."""
        script = os.path.join(
            self.script_dir, 'validate_mali_with_spec.py')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            temp_json = f.name

        try:
            cmd = [
                'python', script,
                '--maliFile', output_file,
                '--specTiff', spec_tiff,
                '--ba_threshold', str(ba_threshold),
                '--output_json', temp_json
            ]

            if plot:
                cmd.append('--plot')
                if plot_dir is not None:
                    cmd += ['--plot_dir', plot_dir]

            result = subprocess.run(cmd, capture_output=True,
                                    text=True)

            if result.returncode == 0 and os.path.exists(temp_json):
                with open(temp_json, 'r') as f:
                    return json.load(f)
            else:
                return {'status': 'failed', 'error': result.stderr}

        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)

    @staticmethod
    def _print_summary(summary, logger):
        """Print analysis summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("ENSEMBLE ANALYSIS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total runs: {summary['total_runs']}")
        logger.info(f"  Completed: {summary['completed_runs']}")
        logger.info(f"  Incomplete: {summary['incomplete_runs']}")
        logger.info("")

        if summary['completed_runs'] > 0:
            tot_runs = summary['completed_runs']
            ss_runs = len(summary['steady_state_runs'])
            dc_runs = len(summary['data_compatible_runs'])
            both_runs = len(summary['both_criteria_runs'])

            pct_ss = 100.0 * ss_runs / tot_runs
            pct_dc = 100.0 * dc_runs / tot_runs
            pct_both = 100.0 * both_runs / tot_runs

            logger.info(
                f"Steady-state runs: {ss_runs}/{tot_runs} "
                f"({pct_ss:.1f}%)")
            if summary['steady_state_runs']:
                logger.info(f"  {summary['steady_state_runs']}")
            logger.info("")
            logger.info(
                f"Data-compatible runs: {dc_runs}/{tot_runs} "
                f"({pct_dc:.1f}%)")
            if summary['data_compatible_runs']:
                logger.info(f"  {summary['data_compatible_runs']}")
            logger.info("")
            logger.info(
                f"Both criteria met: {both_runs}/{tot_runs} "
                f"({pct_both:.1f}%)")
            if summary['both_criteria_runs']:
                logger.info(f"  {summary['both_criteria_runs']}")
            logger.info("")
            logger.info(
                f"Runs needing restart: "
                f"{len(summary['restart_needed_runs'])}")
            if summary['restart_needed_runs']:
                logger.info(f"  {summary['restart_needed_runs']}")

        logger.info("=" * 70)
        logger.info("")
