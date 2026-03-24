"""
Analysis step that performs the actual ensemble analysis.
"""

import configparser
import glob
import json
import os
import subprocess
import tempfile
from datetime import datetime

from compass.step import Step


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

        config_file : str
            Path to configuration file for analysis
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

        if self.config_file is None:
            raise FileNotFoundError(
                f"Could not find ensemble config file for "
                f"{self.ensemble_dir}"
            )

        logger.info(f"Using config file: {self.config_file}")

        # Load configurations
        config_dict = self._load_config(self.config_file)

        # Get analysis configs with defaults
        analysis_config = {
            'steady_state': self._merge_config(
                config_dict.get('steady_state', {}),
                self._get_default_steady_state_config()
            ),
            'validation': self._merge_config(
                config_dict.get('validation', {}),
                self._get_default_validation_config()
            ),
        }

        logger.info(f"Loaded steady_state config: \
                {analysis_config['steady_state']}")
        logger.info(f"Loaded validation config: \
                {analysis_config['validation']}")

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
            results = self._run_analysis_on_run(
                run_dir, run_name, analysis_config)
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
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

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

    def _run_analysis_on_run(self, run_dir, run_name,
                             analysis_config):
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
                plot_results)
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
                output_hist = os.path.join(run_dir, 'output',
                                           'history.nc')
                if os.path.exists(output_hist):
                    val_results = self._run_validation_analysis(
                        output_hist, spec_tiff, ba_threshold,
                        plot_validation)
                    results['validation'] = val_results
                else:
                    results['validation'] = {
                        'status': 'no_history_file'}
            else:
                results['validation'] = {'status': 'no_spec_file',
                                         'spec_tiff': spec_tiff}

        except Exception as e:
            results['analysis_errors'].append(
                f"Validation analysis failed: {e}")
            self.logger.warning(f"    {e}")

        return results

    def _run_steadystate_analysis(self, output_file, window_years,
                                  imbalance_threshold, plot=False):
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

            result = subprocess.run(cmd, capture_output=True,
                                    text=True, timeout=300)

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
                                 ba_threshold, plot=False):
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

            result = subprocess.run(cmd, capture_output=True,
                                    text=True, timeout=300)

            if result.returncode == 0 and os.path.exists(temp_json):
                with open(temp_json, 'r') as f:
                    return json.load(f)
            else:
                return {'status': 'failed', 'error': result.stderr}

        finally:
            if os.path.exists(temp_json):
                os.unlink(temp_json)

    @staticmethod
    def _load_config(config_file):
        """Load configuration file."""
        config = configparser.ConfigParser()
        config.read(config_file)

        config_dict = {}
        for section in config.sections():
            config_dict[section] = {}
            for key, value in config.items(section):
                try:
                    config_dict[section][key] = float(value)
                except ValueError:
                    try:
                        config_dict[section][key] = (
                            config.getboolean(section, key))
                    except ValueError:
                        if value.lower() == 'none':
                            config_dict[section][key] = None
                        else:
                            config_dict[section][key] = value

        return config_dict

    @staticmethod
    def _merge_config(user_config, defaults):
        """
        Merge user config with defaults.
        User config values take precedence.

        Parameters
        ----------
        user_config : dict
            User-provided configuration
        defaults : dict
            Default configuration values

        Returns
        -------
        dict
            Merged configuration
        """
        merged = defaults.copy()
        merged.update(user_config)
        return merged

    @staticmethod
    def _get_default_steady_state_config():
        """Get default steady-state configuration."""
        return {
            'window_years': 10.0,
            'imbalance_threshold': 0.05,
            'plot_results': False,
        }

    @staticmethod
    def _get_default_validation_config():
        """Get default validation configuration."""
        return {
            'balanced_accuracy_threshold': 0.65,
            'spec_tiff_file': None,
            'plot_validation': False,
        }

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
