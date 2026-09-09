"""
Tests for the ISMIP7 calibration config helpers.
"""

import numpy as np
import pytest

from compass.config import CompassConfigParser
from compass.landice.tests.ismip7_calibration.configure import (
    check_options,
    melt_forms,
    objective_options,
    parameter_name,
    parameter_values,
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
    The default K grid must be the 120 values the protocol's worked example
    samples, or the replicated percentiles land on different grid points.
    """
    values = parameter_values(_config(), 'ismip7')

    assert values[0] == pytest.approx(0.25e-5)
    assert values[-1] == pytest.approx(3.0e-4)
    assert len(values) == 120
    np.testing.assert_allclose(np.diff(values), 0.25e-5, rtol=1.0e-9)


def test_parameter_grid_for_the_nonlocal_form():
    """gamma0 has its own grid, in m/yr."""
    values = parameter_values(_config(), 'ismip6')

    assert values[0] == pytest.approx(250.0)
    assert values[-1] == pytest.approx(30000.0)


def test_parameter_values_rejects_an_unknown_form():
    with pytest.raises(ValueError, match="must be 'ismip7' or 'ismip6'"):
        parameter_values(_config(), 'pico')


def test_parameter_name_per_form():
    """The two forms calibrate different parameters."""
    assert parameter_name('ismip7') == 'K'
    assert parameter_name('ismip6') == 'gamma0'
    with pytest.raises(ValueError):
        parameter_name('pico')


def test_melt_forms_parses_a_comma_separated_list():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', 'ismip7, ismip6')

    assert melt_forms(config) == ['ismip7', 'ismip6']


def test_melt_forms_accepts_a_single_form():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', 'ismip7')

    assert melt_forms(config) == ['ismip7']


def test_melt_forms_rejects_an_unknown_form():
    config = _config()
    config.set('ismip7_calibration', 'melt_forms', 'ismip7, pico')

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
