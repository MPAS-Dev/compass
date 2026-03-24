"""
SGH Ensemble Restart Package

Provides test case and scheduling for restarting incomplete ensemble members.

This module identifies runs from a spinup_ensemble that did not complete
or reach steady state, and continues them from their last checkpoint.

Usage:
    compass setup -t landice/ensemble_generator/sgh_restart_ensemble
        -w /work/restart -f restart_ensemble.cfg
    compass run -w /work/restart
"""

from .restart_scheduler import RestartScheduler
from .test_case import RestartEnsemble

__all__ = [
    'RestartEnsemble',
    'RestartScheduler',
]
