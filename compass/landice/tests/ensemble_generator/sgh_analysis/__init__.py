"""
SGH Ensemble Analysis Package.

Provides analysis of completed ensemble runs as a proper compass test case:
- Steady-state detection from water mass balance
- Data compatibility validation against observations
- Results aggregation across ensemble iterations

Usage:
    compass setup -t landice/ensemble_generator/sgh_ensemble_analysis \\
        -w /work/analysis -f analysis_ensemble.cfg
    compass run -w /work/analysis
"""

from .results_aggregator import ResultsAggregator
from .test_case import AnalysisEnsemble

__all__ = [
    'AnalysisEnsemble',
    'ResultsAggregator',
]
