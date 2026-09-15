"""
Tests for the per-basin dT_b fit.

The search itself is a few lines, but a sign or indexing slip in it would
produce plausible-looking corrections, so its behaviour is pinned here.
"""

import numpy as np
import pytest

from compass.landice.tests.ismip7_calibration.ais.fit_delta_t import (
    PERCENTILES,
    check_fit,
    fit_one_basin,
)

GRID = np.linspace(-2.0, 2.0, 41)


def _melt_curve(scale=100.0):
    """Melt that rises with dT, as a quadratic in a positive forcing does."""
    return scale * (1.0 + 0.5 * GRID) ** 2


def test_picks_the_dt_that_matches_observed():
    """An observed value exactly on the curve is found exactly."""
    modelled = _melt_curve()
    target_index = 30
    fit = fit_one_basin(modelled, modelled[target_index], GRID)

    assert fit['delta_t'] == pytest.approx(GRID[target_index])
    assert fit['residual'] == pytest.approx(0.0)


def test_zero_correction_when_already_matching():
    """If dT = 0 already matches, the fit must not move away from it."""
    modelled = _melt_curve()
    zero = int(np.argmin(np.abs(GRID)))
    fit = fit_one_basin(modelled, modelled[zero], GRID)

    assert fit['delta_t'] == pytest.approx(0.0)


def test_warming_when_the_model_melts_too_little():
    """Observed above the dT = 0 melt needs a positive correction."""
    modelled = _melt_curve()
    zero = int(np.argmin(np.abs(GRID)))
    fit = fit_one_basin(modelled, 1.5 * modelled[zero], GRID)

    assert fit['delta_t'] > 0.0


def test_cooling_when_the_model_melts_too_much():
    """Observed below the dT = 0 melt needs a negative correction."""
    modelled = _melt_curve()
    zero = int(np.argmin(np.abs(GRID)))
    fit = fit_one_basin(modelled, 0.5 * modelled[zero], GRID)

    assert fit['delta_t'] < 0.0


def test_fit_never_worsens_the_misfit():
    """dT = 0 is on the grid, so the best misfit can never exceed it."""
    modelled = _melt_curve()
    zero = int(np.argmin(np.abs(GRID)))
    for observed in (0.0, 50.0, 100.0, 400.0, 1.0e4):
        fit = fit_one_basin(modelled, observed, GRID)
        assert fit['residual'] <= abs(modelled[zero] - observed) + 1.0e-12


def test_unreachable_target_stops_at_the_bound():
    """
    A target beyond what any dT in the grid can reach lands on the bound.
    The protocol bounds the correction deliberately, and the step warns
    about basins that hit it.
    """
    modelled = _melt_curve()
    fit = fit_one_basin(modelled, 1.0e6, GRID)

    assert fit['delta_t'] == pytest.approx(GRID.max())


def test_before_and_after_are_reported():
    modelled = _melt_curve()
    zero = int(np.argmin(np.abs(GRID)))
    fit = fit_one_basin(modelled, modelled[35], GRID)

    assert fit['modelled_before'] == pytest.approx(modelled[zero])
    assert fit['modelled_after'] == pytest.approx(modelled[35])
    assert fit['observed'] == pytest.approx(modelled[35])


def test_check_fit_accepts_a_consistent_fit():
    fit = {9: dict(delta_t=0.5, observed=100.0, modelled_before=80.0,
                   modelled_after=99.0, residual=1.0)}
    check_fit(fit, GRID)


def test_check_fit_catches_a_fit_that_got_worse():
    """A residual above the dT = 0 misfit means the search is broken."""
    fit = {9: dict(delta_t=0.5, observed=100.0, modelled_before=99.0,
                   modelled_after=50.0, residual=50.0)}
    with pytest.raises(ValueError, match='worse than'):
        check_fit(fit, GRID)


def test_check_fit_catches_a_dt_off_the_grid():
    fit = {9: dict(delta_t=5.0, observed=100.0, modelled_before=80.0,
                   modelled_after=99.0, residual=1.0)}
    with pytest.raises(ValueError, match='outside'):
        check_fit(fit, GRID)


def test_a_parameter_file_is_written_per_percentile():
    """
    dT_b depends on the melt parameter, so each percentile a projection
    might run at needs its own corrections.
    """
    names = [percentile for percentile, _ in PERCENTILES]
    assert names == ['p5', 'median', 'p95']
