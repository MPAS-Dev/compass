"""
Shared configuration checks for the ISMIP7 calibration test cases.
"""

import numpy as np

from compass.landice.tests.ismip7_calibration.datasets import (
    PUBLISHED_T3_MODELS,
    PUBLISHED_T4_REGIONS,
    PUBLISHED_T4_YEARS,
)

#: The two ISMIP7 melt forms: constant slope and spatially-varying slope
ISMIP7_FORMS = ('ismip7_const', 'ismip7_slope')

#: All valid melt forms
ALL_FORMS = ISMIP7_FORMS + ('ismip6',)


def is_ismip7(melt_form):
    """
    Check if a melt form is one of the ISMIP7 family (both const and slope).

    Parameters
    ----------
    melt_form : str
        The melt form name

    Returns
    -------
    bool
        True if the form is an ISMIP7 variant (uses the K parameter)
    """
    return melt_form in ISMIP7_FORMS


def uses_spatial_slope(melt_form):
    """
    Check if a melt form uses spatially-varying shelf-base slope.

    Parameters
    ----------
    melt_form : str
        The melt form name

    Returns
    -------
    bool
        True if the form uses MALI's diagnosed per-cell slope field
    """
    return melt_form == 'ismip7_slope'


def mali_melt_method(melt_form):
    """
    Map a melt form name to MALI's config_basal_mass_bal_float value.

    Parameters
    ----------
    melt_form : str
        The melt form name

    Returns
    -------
    str
        The value for MALI's config_basal_mass_bal_float namelist option
    """
    return 'ismip7' if is_ismip7(melt_form) else 'ismip6'


def check_options(config, options):
    """
    Check that user-supplied config options have been set.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    options : list of str
        Options in the ``ismip7_calibration`` section that the user must
        supply

    Raises
    ------
    ValueError
        If any of ``options`` is still ``NotAvailable``
    """
    section = 'ismip7_calibration'
    for option in options:
        value = config.get(section=section, option=option)
        if value == 'NotAvailable':
            raise ValueError(
                f'You need to supply a user config file containing the '
                f'[{section}] section with the {option} option set.')


def parameter_values(config, melt_form):
    """
    The grid of parameter values to select from, for one melt form.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    melt_form : {'ismip7_const', 'ismip7_slope', 'ismip6'}
        Which melt form the parameter belongs to.  ``'ismip7_const'`` and
        ``'ismip7_slope'`` calibrate ``K``; ``'ismip6'`` calibrates
        ``gamma0``.

    Returns
    -------
    values : numpy.ndarray
        The parameter grid
    """
    section = config['ismip7_calibration_melt']
    if melt_form == 'ismip7_const':
        low = section.getfloat('k_min')
        high = section.getfloat('k_max')
        step = section.getfloat('k_step')
    elif uses_spatial_slope(melt_form):
        low = section.getfloat('slope_k_min')
        high = section.getfloat('slope_k_max')
        step = section.getfloat('slope_k_step')
    elif melt_form == 'ismip6':
        low = section.getfloat('gamma0_min')
        high = section.getfloat('gamma0_max')
        step = section.getfloat('gamma0_step')
    else:
        raise ValueError(f"melt_form must be one of {ALL_FORMS}, but is "
                         f"'{melt_form}'")

    # add half a step so that ``high`` itself is included
    return np.arange(low, high + 0.5 * step, step)


def parameter_name(melt_form):
    """
    The name of the parameter a melt form calibrates.

    Parameters
    ----------
    melt_form : {'ismip7_const', 'ismip7_slope', 'ismip6'}
        The melt form

    Returns
    -------
    name : str
        ``'K'`` for the ISMIP7 forms and ``'gamma0'`` for ``'ismip6'``
    """
    if is_ismip7(melt_form):
        return 'K'
    if melt_form == 'ismip6':
        return 'gamma0'
    raise ValueError(f"melt_form must be one of {ALL_FORMS}, but is "
                     f"'{melt_form}'")


def melt_forms(config):
    """
    The melt forms to calibrate.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    Returns
    -------
    forms : list of str
        Each of ``'ismip7_const'``, ``'ismip7_slope'``, and ``'ismip6'``
        that was requested
    """
    value = config.get('ismip7_calibration', 'melt_forms')
    forms = [form.strip() for form in value.split(',') if form.strip()]
    for form in forms:
        if form not in ALL_FORMS:
            raise ValueError(f"melt_forms must contain only {ALL_FORMS}, "
                             f"but contains '{form}'")
    if not forms:
        raise ValueError('melt_forms must name at least one melt form')
    return forms


def weighting(config):
    """
    Which summands of J3 and J4 carry non-zero weight.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    Returns
    -------
    t3_models : tuple of str or None
        Ocean models to weight in J3, or None to weight all of them

    t4_regions : tuple of str or None
        Ice shelves to weight in J4, or None to weight all of them

    t4_years : tuple of int or None
        Observation years to weight in J4, or None to weight all of them
    """
    section = config['ismip7_calibration_objective']

    def _select(option, published):
        value = section.get(option).strip()
        if value == 'published':
            return published
        if value == 'all':
            return None
        raise ValueError(f"{option} must be 'published' or 'all', but is "
                         f"'{value}'")

    return (_select('t3_models', PUBLISHED_T3_MODELS),
            _select('t4_regions', PUBLISHED_T4_REGIONS),
            _select('t4_years', PUBLISHED_T4_YEARS))


def objective_options(config):
    """
    The sample size and random seed for the parameter selection.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    Returns
    -------
    sample_size : int
        Number of random draws

    seed : int or None
        Seed for the random draws, or None to leave the random state alone
    """
    section = config['ismip7_calibration_objective']
    sample_size = section.getint('sample_size')
    seed = section.get('seed').strip()
    seed = None if seed.lower() == 'none' else int(seed)
    return sample_size, seed
