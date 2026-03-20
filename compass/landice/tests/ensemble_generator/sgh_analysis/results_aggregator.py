"""
Aggregate results across multiple ensemble iterations.
"""

import glob
import json
import os


class ResultsAggregator:
    """
    Combine results from multiple ensemble iterations (initial + restarts).
    """

    def __init__(self, base_dir):
        """
        Initialize aggregator.

        Parameters
        ----------
        base_dir : str
            Parent directory containing analysis work directories
            (where you ran compass setup/run for analysis_ensemble)
        """
        self.base_dir = base_dir

    def find_summary_files(self):
        """
        Find all analysis_summary.json files.

        Searches for:
        - /analysis_ensemble1/analysis_summary.json
        - /analysis_ensemble2/analysis_summary.json
        etc.

        Returns
        -------
        list of str
            Paths to summary files, sorted
        """
        # Look in subdirectories (analysis work dirs)
        summaries = glob.glob(
            os.path.join(
                self.base_dir,
                '*/analysis_summary.json'))
        return sorted(summaries)

    def aggregate(self):
        """
        Aggregate results from all analysis iterations.

        Returns
        -------
        dict
            Aggregated results
        """
        summaries = self.find_summary_files()

        if not summaries:
            print("No summary files found")
            return None

        aggregated = {
            'iterations': [],
            'total_completed': 0,
            'total_steady_state': 0,
            'total_data_compatible': 0,
            'total_both_criteria': 0,
            'final_steady_state_runs': [],
            'final_data_compatible_runs': [],
            'final_both_criteria_runs': [],
        }

        all_steady = set()
        all_compatible = set()
        all_both = set()

        for summary_file in summaries:
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            iteration = {
                'timestamp': summary['timestamp'],
                'ensemble_dir': summary['ensemble_dir'],
                'completed': summary['completed_runs'],
                'steady_state': len(summary['steady_state_runs']),
                'data_compatible': len(summary['data_compatible_runs']),
                'both_criteria': len(summary['both_criteria_runs']),
            }

            aggregated['iterations'].append(iteration)
            aggregated['total_completed'] += summary['completed_runs']
            aggregated['total_steady_state'] += len(
                summary['steady_state_runs'])
            aggregated['total_data_compatible'] += len(
                summary['data_compatible_runs'])
            aggregated['total_both_criteria'] += len(
                summary['both_criteria_runs'])

            all_steady.update(summary['steady_state_runs'])
            all_compatible.update(summary['data_compatible_runs'])
            all_both.update(summary['both_criteria_runs'])

        aggregated['final_steady_state_runs'] = sorted(list(all_steady))
        aggregated['final_data_compatible_runs'] = sorted(list(all_compatible))
        aggregated['final_both_criteria_runs'] = sorted(list(all_both))

        return aggregated

    def print_summary(self, aggregated):
        """Print aggregated summary."""
        print("\n" + "=" * 70)
        print("ENSEMBLE AGGREGATED RESULTS")
        print("=" * 70)

        for i, it in enumerate(aggregated['iterations'], 1):
            print(f"\nIteration {i}:")
            print(f"  Ensemble: {it['ensemble_dir']}")
            print(f"  Completed: {it['completed']}")
            print(f"  Steady-state: {it['steady_state']}")
            print(f"  Data-compatible: {it['data_compatible']}")
            print(f"  Both criteria: {it['both_criteria']}")

        print("\nFinal Results (across all iterations):")
        print(f"  Total completed: {aggregated['total_completed']}")
        print(
            f"Steady-state runs: {len(aggregated['final_steady_state_runs'])}")
        print(f"    {aggregated['final_steady_state_runs']}")
        print(
            f"Data-compatible runs: \
                    {len(aggregated['final_data_compatible_runs'])}")
        print(f"    {aggregated['final_data_compatible_runs']}")
        print(
            f"Both criteria: {len(aggregated['final_both_criteria_runs'])}")
        print(f"{aggregated['final_both_criteria_runs']}")
        print("=" * 70 + "\n")

    def save_aggregated(self, aggregated, filename='aggregated_results.json'):
        """
        Save aggregated results.

        Parameters
        ----------
        aggregated : dict
            Aggregated results dictionary

        filename : str
            Output filename
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"Aggregated results saved to {filepath}")
