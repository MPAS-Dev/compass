"""
Tests for the ISMIP7 calibration config helpers.
"""

import numpy as np
import pytest

from compass.config import CompassConfigParser
from compass.landice.tests.ismip7_calibration.configure import (
    ALL_FORMS,
    ISMIP7_FORMS,
    check_options,
    is_ismip7,
    mali_melt_method,
    melt_forms,
    objective_options,
    parameter_name,
    parameter_values,
    uses_spatial_slope,
    weighting,
)


def _config():
    """The test group's default config options."""
    config = CompassConfigParser()
    config.add_from_package('compass.landice.tests.ismip7_calibration',
                            'ismip7_calibration.cfg')
    return config


def test_check_options_rejects_unset_paths():
    """A path the user must supply should fail loudly, not silently."""
    config = _config()

    with pytest.raises(ValueError, match='base_path_ismip7'):
        check_options(config, ['base_path_ismip7'])


def test_check_options_accepts_a_supplied_path():
    """Once set, the same option passes."""
    config = _config()
    config.set('ismip7_calibration', 'base_path_ismip7', '/some/path')

    check_options(config, ['base_path_ismip7'])


def test_parameter_grid_matches_the_published_k_values():
    """
    The default K grid for ismip7_const must be the 120 values the protocol's
    worked example samples, or the replicated percentiles land on different
    grid points.
    """
    values = parameter_values(_config(), 'ismip7_const')

    assert values[0] == pytest.approx(0.25e-5)
    assert values[-1] == pytest.approx(3.0e-4)
    assert len(values) == 120
    np.testing.assert_allclose(np.diff(values), 0.25e-5, rtol=1.0e-9)


def test_parameter_grid_for_the_nonlocal_form():
    """
    gamma0 has its own grid, in m/yr.  The maximum has to stay well above
    where the distribution ends, or the objective's draws pile up against
    it and the upper percentiles become a property of the grid.
    """
    values = parameter_values(_config(), 'ismip6')

    assert values[0] == pytest.approx(250.0)
    assert values[-1] == pytest.approx(50000.0)


def test_parameter_values_rejects_an_unknown_form():
    with pytest.raises(ValueError, match='pico'):
        parameter_values(_config(), 'pico')


def test_melt_forms_parses_a_comma_separated_list():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms',
               'ismip7_const, ismip6')

    assert melt_forms(config) == ['ismip7_const', 'ismip6']


def test_melt_forms_accepts_a_single_form():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', 'ismip7_const')

    assert melt_forms(config) == ['ismip7_const']


def test_melt_forms_rejects_an_unknown_form():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', 'ismip7_const, pico')

    with pytest.raises(ValueError, match='pico'):
        melt_forms(config)


def test_melt_forms_rejects_an_empty_list():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', '')

    with pytest.raises(ValueError, match='at least one'):
        melt_forms(config)


def test_weighting_all_gives_no_restriction():
    """'all' means every summand the ensemble provides carries weight."""
    config = _config()
    config.set('ismip7_calibration_objective', 't3_models', 'all')
    config.set('ismip7_calibration_objective', 't4_regions', 'all')
    config.set('ismip7_calibration_objective', 't4_years', 'all')

    assert weighting(config) == (None, None, None)


def test_weighting_published_restricts_j4_to_pig():
    """
    The published weighting uses PIG alone in 2009 and 2012 -- 2 of the 18
    available observations.  That is a deliberate option, not the default,
    because so weighted J4 cannot discriminate between melt forms.
    """
    config = _config()
    config.set('ismip7_calibration_objective', 't3_models', 'published')
    config.set('ismip7_calibration_objective', 't4_regions', 'published')
    config.set('ismip7_calibration_objective', 't4_years', 'published')

    t3_models, t4_regions, t4_years = weighting(config)

    assert t4_regions == ('pig',)
    assert t4_years == (2009, 2012)
    assert len(t3_models) == 4


def test_weighting_rejects_an_unknown_choice():
    config = _config()
    config.set('ismip7_calibration_objective', 't3_models', 'some')

    with pytest.raises(ValueError, match="must be 'published' or 'all'"):
        weighting(config)


def test_objective_options_defaults_are_reproducible():
    """A fixed seed is the default, so the percentiles do not drift."""
    sample_size, seed = objective_options(_config())

    assert sample_size == 100000
    assert seed == 0


def test_objective_options_allows_no_seed():
    """'None' leaves the global random state alone."""
    config = _config()
    config.set('ismip7_calibration_objective', 'seed', 'None')

    _, seed = objective_options(config)

    assert seed is None


def test_parameter_grid_for_ismip7_slope():
    """
    The ismip7_slope form uses a finer K grid than ismip7_const, since the
    spatial-slope p5-to-p95 span is narrower.
    """
    config = _config()
    values = parameter_values(config, 'ismip7_slope')

    assert values[0] == pytest.approx(0.05e-5)
    assert values[-1] == pytest.approx(1.0e-4)
    assert len(values) == 200  # (1.0e-4 - 0.05e-5) / 0.05e-5 + 1 = 199.5 + 1
    np.testing.assert_allclose(np.diff(values), 0.05e-5, rtol=1.0e-9)


def test_parameter_name_for_all_forms():
    """Both ISMIP7 forms return 'K', ISMIP6 returns 'gamma0'."""
    assert parameter_name('ismip7_const') == 'K'
    assert parameter_name('ismip7_slope') == 'K'
    assert parameter_name('ismip6') == 'gamma0'


def test_parameter_name_rejects_unknown_form():
    with pytest.raises(ValueError, match='pico'):
        parameter_name('pico')


def test_is_ismip7_helper():
    """is_ismip7 returns True for both ISMIP7 forms."""
    assert is_ismip7('ismip7_const')
    assert is_ismip7('ismip7_slope')
    assert not is_ismip7('ismip6')


def test_uses_spatial_slope_helper():
    """uses_spatial_slope returns True only for ismip7_slope."""
    assert not uses_spatial_slope('ismip7_const')
    assert uses_spatial_slope('ismip7_slope')
    assert not uses_spatial_slope('ismip6')


def test_mali_melt_method_helper():
    """
    mali_melt_method maps form names to MALI's config_basal_mass_bal_float
    value. Both ISMIP7 forms map to 'ismip7'.
    """
    assert mali_melt_method('ismip7_const') == 'ismip7'
    assert mali_melt_method('ismip7_slope') == 'ismip7'
    assert mali_melt_method('ismip6') == 'ismip6'


def test_melt_forms_accepts_all_three_forms():
    """The three valid forms can be listed in any combination."""
    config = _config()
    config.set('ismip7_calibration', 'melt_forms',
               'ismip7_const, ismip7_slope, ismip6')
    forms = melt_forms(config)
    assert forms == ['ismip7_const', 'ismip7_slope', 'ismip6']

    config.set('ismip7_calibration', 'melt_forms', 'ismip7_slope')
    forms = melt_forms(config)
    assert forms == ['ismip7_slope']


def test_constants_match_helpers():
    """The ISMIP7_FORMS and ALL_FORMS constants are consistent."""
    assert ISMIP7_FORMS == ('ismip7_const', 'ismip7_slope')
    assert ALL_FORMS == ('ismip7_const', 'ismip7_slope', 'ismip6')
    for form in ISMIP7_FORMS:
        assert is_ismip7(form)
    for form in ALL_FORMS:
        assert form in ALL_FORMS
