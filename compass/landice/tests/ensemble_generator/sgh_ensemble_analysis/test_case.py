"""
Analysis ensemble test case for SGH template.

Analyzes a completed ensemble run (spinup or restart) and produces
summary statistics and visualizations.

Usage:
    compass setup -t landice/ensemble_generator/sgh_ensemble_analysis \\
        -w /work/analysis -f analysis_ensemble.cfg
    compass run -w /work/analysis
"""

import os

from compass.testcase import TestCase

from .analysis_step import AnalysisStep


class AnalysisEnsemble(TestCase):
    """
    A test case for analyzing completed ensemble runs.

    This test case:
    1. Reads a completed ensemble directory
    2. Analyzes each run for steady-state and data compatibility
    3. Generates analysis_summary.json with results
    """

    def __init__(self, test_group):
        """
        Create the analysis ensemble test case.

        Parameters
        ----------
        test_group : compass test group
            The test group that this test case belongs to
        """
        name = 'sgh_ensemble_analysis'
        super().__init__(test_group=test_group, name=name)

    def configure(self):
        """
        Configure analysis by reading ensemble directory to analyze.
        """
        config = self.config

        try:
            ensemble_dir = config.get('analysis_ensemble',
                                      'ensemble_work_dir')
        except Exception:
            raise ValueError(
                "analysis_ensemble config must specify:\n"
                "  ensemble_work_dir\n"
                "Add to config file:\n"
                "[analysis_ensemble]\n"
                "ensemble_work_dir = /path/to/ensemble/work/dir\n"
                "ensemble_config_file = /path/to/ensemble.cfg"
            )

        if not os.path.exists(ensemble_dir):
            raise ValueError(
                f"ensemble_work_dir not found: {ensemble_dir}"
            )

        # Add single analysis step
        self.add_step(AnalysisStep(
            test_case=self,
            ensemble_dir=ensemble_dir
        ))
