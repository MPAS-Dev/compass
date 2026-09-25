"""
Tests for run_state.py helpers.
"""

import pytest

from compass.config import CompassConfigParser
from compass.landice.tests.ismip7_calibration.ais.run_state import (
    PARAMETER_VARIABLE,
    REQUIRED_OPTIONS,
    _melt_namelist_options,
    reference_parameter,
)


def _config():
    """The test group's default config options."""
    config = CompassConfigParser()
    config.add_from_package('compass.landice.tests.ismip7_calibration',
                            'ismip7_calibration.cfg')
    return config


def test_required_options_ismip7_const_omits_slope():
    """
    ismip7_const requires only the base ISMIP7 options, not the PR #195
    slope options, so it can run on an older MALI build.
    """
    required = REQUIRED_OPTIONS['ismip7_const']
    assert 'config_ismip7_melt_sin_slope' in required
    assert 'config_ismip7_melt_coriolis' in required
    assert 'config_ismip7_melt_salinity' in required
    assert 'config_basal_mass_bal_float' in required
    # Slope options should NOT be required
    assert 'config_ismip7_melt_spatially_variable_slope' not in required
    assert 'config_ismip7_melt_max_slope' not in required


def test_required_options_ismip7_slope_includes_slope():
    """
    ismip7_slope requires the full ISMIP7 option set including the five
    slope options from PR #195.
    """
    required = REQUIRED_OPTIONS['ismip7_slope']
    # Base ISMIP7 options
    assert 'config_ismip7_melt_sin_slope' in required
    assert 'config_ismip7_melt_coriolis' in required
    assert 'config_basal_mass_bal_float' in required
    # Slope options
    assert 'config_ismip7_melt_spatially_variable_slope' in required
    assert 'config_ismip7_melt_max_slope' in required
    assert 'config_ismip7_melt_slope_smoothing_iterations' in required
    assert 'config_ismip7_melt_slope_method' in required
    assert 'config_ismip7_melt_slope_stencil_rings' in required


def test_required_options_ismip6():
    """ISMIP6 only needs config_basal_mass_bal_float."""
    required = REQUIRED_OPTIONS['ismip6']
    assert required == ['config_basal_mass_bal_float']


def test_parameter_variable_mapping():
    """Both ISMIP7 forms map to ismip7shelfMelt_K."""
    assert PARAMETER_VARIABLE['ismip7_const'] == 'ismip7shelfMelt_K'
    assert PARAMETER_VARIABLE['ismip7_slope'] == 'ismip7shelfMelt_K'
    assert PARAMETER_VARIABLE['ismip6'] == 'ismip6shelfMelt_gamma0'


def test_melt_namelist_options_ismip7_const():
    """
    ismip7_const emits base ISMIP7 options plus
    spatially_variable_slope = .false., but NOT the slope configuration.
    """
    config = _config()
    opts = _melt_namelist_options(config, 'ismip7_const')

    assert opts['config_ismip7_melt_sin_slope'] == repr(0.0051117)
    assert opts['config_ismip7_melt_coriolis'] == repr(1.4e-4)
    assert opts['config_ismip7_melt_salinity_source'] == "'constant'"
    assert opts['config_ismip7_melt_salinity'] == repr(34.5)
    assert opts['config_ismip7_melt_spatially_variable_slope'] == '.false.'
    # Slope options should NOT be emitted
    assert 'config_ismip7_melt_max_slope' not in opts
    assert 'config_ismip7_melt_slope_method' not in opts


def test_melt_namelist_options_ismip7_slope():
    """
    ismip7_slope emits base ISMIP7 options plus spatially_variable_slope
    = .true. and all five slope configuration options.
    """
    config = _config()
    opts = _melt_namelist_options(config, 'ismip7_slope')

    # Base options
    assert opts['config_ismip7_melt_sin_slope'] == repr(0.0051117)
    assert opts['config_ismip7_melt_coriolis'] == repr(1.4e-4)
    assert opts['config_ismip7_melt_salinity_source'] == "'constant'"
    assert opts['config_ismip7_melt_salinity'] == repr(34.5)
    # Slope enabled
    assert opts['config_ismip7_melt_spatially_variable_slope'] == '.true.'
    # Slope configuration
    assert opts['config_ismip7_melt_max_slope'] == repr(0.5)
    assert opts['config_ismip7_melt_slope_smoothing_iterations'] == repr(1)
    assert opts['config_ismip7_melt_slope_method'] == "'local'"
    assert opts['config_ismip7_melt_slope_stencil_rings'] == repr(3)


def test_melt_namelist_options_ismip6():
    """ISMIP6 emits no melt-specific options (gamma0 is file-read)."""
    config = _config()
    opts = _melt_namelist_options(config, 'ismip6')

    assert opts == {}


def test_reference_parameter_for_ismip7_forms():
    """Both ISMIP7 forms return reference_k."""
    config = _config()
    assert reference_parameter(config, 'ismip7_const') == pytest.approx(1.0e-4)
    assert reference_parameter(config, 'ismip7_slope') == pytest.approx(1.0e-4)


def test_reference_parameter_for_ismip6():
    """ISMIP6 returns reference_gamma0."""
    config = _config()
    assert reference_parameter(config, 'ismip6') == pytest.approx(10000.0)
