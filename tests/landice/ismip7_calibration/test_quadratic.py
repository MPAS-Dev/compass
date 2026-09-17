"""
Tests for the Python reference implementations of the melt formulas.

These are the specification MALI's Fortran is checked against, so the
properties the calibration depends on are asserted here directly.
"""

import numpy as np
import pytest
import xarray as xr

from compass.landice.tests.ismip7_calibration.quadratic import (
    MALI,
    MULTIMELT,
    angle_from_sin_slope,
    local_quadratic_melt,
    nonlocal_quadratic_melt,
)

SALINITY = 34.5
SLOPE = angle_from_sin_slope(0.0051117)


def _thermal_forcing(values):
    return xr.DataArray(np.asarray(values, dtype=float), dims='nCells')


def test_angle_from_sin_slope_round_trips():
    """
    MALI's config option is a sine while local_quadratic_melt takes an
    angle; mixing them would silently rescale the melt.
    """
    for sin_slope in (0.0051117, 0.005, 0.1):
        assert np.sin(angle_from_sin_slope(sin_slope)) == \
            pytest.approx(sin_slope)


def test_local_melt_is_exactly_linear_in_k():
    """
    The whole ensemble design rests on this: one MALI run per ocean state,
    scaled onto the parameter grid, instead of one run per (state, K) pair.
    """
    tf = _thermal_forcing([0.5, 1.0, 2.0, -0.5])
    base = local_quadratic_melt(1.0, tf, SALINITY, SLOPE, constants=MALI)

    for factor in (0.25, 3.0, 17.0):
        scaled = local_quadratic_melt(factor, tf, SALINITY, SLOPE,
                                      constants=MALI)
        np.testing.assert_allclose(scaled.values, factor * base.values,
                                   rtol=1.0e-14, atol=0.0)


def test_nonlocal_melt_is_exactly_linear_in_gamma0():
    """The same linearity must hold for the ISMIP6 non-local form."""
    tf = _thermal_forcing([0.5, 1.0, 2.0, -0.5])
    mean_tf = _thermal_forcing([1.0, 1.0, 1.0, 1.0])
    base = nonlocal_quadratic_melt(1.0, tf, mean_tf, constants=MALI)

    for factor in (0.25, 3.0, 17.0):
        scaled = nonlocal_quadratic_melt(factor, tf, mean_tf, constants=MALI)
        np.testing.assert_allclose(scaled.values, factor * base.values,
                                   rtol=1.0e-14, atol=0.0)


def test_local_melt_is_quadratic_in_thermal_forcing():
    """Doubling the thermal forcing must quadruple the melt."""
    single = local_quadratic_melt(1.0, _thermal_forcing([1.0]), SALINITY,
                                  SLOPE, constants=MALI).item()
    double = local_quadratic_melt(1.0, _thermal_forcing([2.0]), SALINITY,
                                  SLOPE, constants=MALI).item()
    assert double == pytest.approx(4.0 * single)


def test_melt_keeps_the_sign_of_the_thermal_forcing():
    """
    The |TF| TF factor means warm water melts and cold water refreezes; a
    sign error here would be invisible in a total but wrong per basin.
    """
    tf = _thermal_forcing([2.0, -2.0])
    melt = local_quadratic_melt(1.0, tf, SALINITY, SLOPE, constants=MALI)
    assert melt[0].item() > 0.0
    assert melt[1].item() < 0.0
    assert melt[0].item() == pytest.approx(-melt[1].item())


def test_local_melt_is_linear_in_salinity():
    """
    Melt is linear in salinity, which is what bounds the effect of using a
    constant value rather than a field.
    """
    tf = _thermal_forcing([1.0])
    low = local_quadratic_melt(1.0, tf, 34.0, SLOPE,
                               constants=MALI).item()
    high = local_quadratic_melt(1.0, tf, 34.0 * 2.0, SLOPE,
                                constants=MALI).item()
    assert high == pytest.approx(2.0 * low)


def test_delta_t_shifts_both_thermal_forcing_factors():
    """
    Protocol Sect. 4.2.1 applies the basin correction wherever the thermal
    forcing appears, so melt at TF with dT must equal melt at TF + dT with
    no correction.
    """
    tf = _thermal_forcing([1.0, 2.0])
    with_correction = local_quadratic_melt(1.0, tf, SALINITY, SLOPE,
                                           constants=MALI, delta_t=0.5)
    shifted = local_quadratic_melt(1.0, tf + 0.5, SALINITY, SLOPE,
                                   constants=MALI)
    np.testing.assert_allclose(with_correction.values, shifted.values,
                               rtol=1.0e-14)


def test_semi_local_is_degenerate_with_the_nonlocal_form():
    """
    With a constant salinity the Burgard semi-local form is algebraically
    identical to the ISMIP6 non-local method MALI already has, so every K
    has an exactly equivalent gamma0.  That degeneracy is why the
    semi-local form is calibrated through the ISMIP6 code path rather than
    added as a third melt module, so it is worth asserting.
    """
    tf = _thermal_forcing([0.5, 1.5, -0.5])
    mean_tf = _thermal_forcing([1.0, 1.0, 1.0])

    melt_k = 8.5e-5
    # semi-local: the local form with the basin mean in the |TF| factor
    semi_local = local_quadratic_melt(melt_k, tf, SALINITY, SLOPE,
                                      thermal_forcing_avg=mean_tf,
                                      constants=MALI)

    # the equivalent gamma0 follows from equating the two constants
    cste = (MALI.rho_ocean * MALI.c_o /
            (MALI.rho_ice * MALI.latent_heat))**2
    # gamma0 is already a per-year velocity scale, so the conversion
    # carries the year length that the local form applies explicitly
    coefficient = (np.sin(SLOPE) * (MALI.rho_ocean / MALI.rho_ice) *
                   (MALI.c_o / MALI.latent_heat)**2 * MALI.beta_s *
                   MALI.gravity / (2.0 * abs(MALI.coriolis)) * SALINITY *
                   MALI.seconds_per_year)
    gamma0 = melt_k * coefficient / cste

    non_local = nonlocal_quadratic_melt(gamma0, tf, mean_tf, constants=MALI)

    np.testing.assert_allclose(semi_local.values, non_local.values,
                               rtol=1.0e-12)


def test_local_and_semi_local_differ_in_pattern():
    """
    Replacing the basin mean with the local forcing changes the spatial
    pattern in a way no per-basin constant can undo.  If these agreed, the
    two forms would be indistinguishable and the comparison pointless.
    """
    tf = _thermal_forcing([0.5, 2.0])
    mean_tf = _thermal_forcing([1.25, 1.25])

    local = local_quadratic_melt(1.0, tf, SALINITY, SLOPE, constants=MALI)
    semi_local = local_quadratic_melt(1.0, tf, SALINITY, SLOPE,
                                      thermal_forcing_avg=mean_tf,
                                      constants=MALI)

    ratio = (local / semi_local).values
    assert not np.allclose(ratio, ratio[0])


def test_year_length_differs_between_the_two_constant_sets():
    """
    MALI uses a 365-day year and the protocol's reference implementation
    365.2422 days.  The 0.066% difference is small but showed up as
    unexplained noise before it was tracked down, so it is pinned here.
    """
    ratio = MULTIMELT.seconds_per_year / MALI.seconds_per_year
    assert ratio == pytest.approx(1.000663562, rel=1.0e-8)


def test_melt_is_independent_of_ice_density_for_mali_constants():
    """
    The ice density appears once in the formula and once in the conversion
    to a mass flux, so for MALI's constants the two cancel exactly.  The
    protocol's worked example uses 917 and 918, so they do not cancel there
    -- which is why both are carried separately.
    """
    assert MALI.rho_ice == MALI.rho_ice_flux
    assert MULTIMELT.rho_ice != MULTIMELT.rho_ice_flux
