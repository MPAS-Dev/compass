"""
Tests for assembling the ISMIP7 objective function's arguments.

The weighting decides which observations constrain the calibration at all,
so getting it wrong changes the answer silently.  These tests pin the
scaling that the one-run-per-state ensemble design relies on, and the
weighting rules.
"""

import numpy as np
import pytest
import xarray as xr

from compass.landice.tests.ismip7_calibration.objective import (
    build_toolbox_terms,
    scale_to_ensemble,
)

PARAMETERS = np.array([1.0, 2.0, 4.0])
BASINS = np.arange(3)
MODELS = ['mathiot', 'haid']
REGIONS = ['pig', 'dotson']
YEARS = [2009, 2012, 2014]


def _units():
    """Unit-parameter aggregates with the shapes the toolbox expects."""
    t1 = xr.DataArray(np.ones(3), dims='basins', coords={'basins': BASINS})
    t2 = xr.DataArray(np.ones(4), dims='BFRN_bins',
                      coords={'BFRN_bins': np.arange(4)})
    t3 = xr.DataArray(np.ones((2, 3)), dims=('model', 'basins'),
                      coords={'model': MODELS, 'basins': BASINS})
    t4 = xr.DataArray(np.ones((3, 2)), dims=('year', 'region'),
                      coords={'year': YEARS, 'region': REGIONS})
    return dict(t1=t1, t2=t2, t3=t3, t4=t4)


def _targets():
    """Targets matching the aggregate shapes."""
    t4_mean = xr.DataArray(np.ones((2, 3)), dims=('region', 'year'),
                           coords={'region': REGIONS, 'year': YEARS})
    return dict(
        t1_mean=xr.DataArray(np.ones(3), dims='basin'),
        t1_sigma=xr.DataArray(np.ones(3), dims='basin'),
        t2_mean=xr.DataArray(np.ones(4), dims='BFRN_bins'),
        t2_sigma=xr.DataArray(np.ones(4), dims='BFRN_bins'),
        t2_weights=xr.DataArray(np.ones(4), dims='BFRN_bins'),
        t3_mean=xr.DataArray(np.ones((2, 3)), dims=('model', 'basins'),
                             coords={'model': MODELS, 'basins': BASINS}),
        t3_sigma=xr.DataArray(np.ones((2, 3)), dims=('model', 'basins'),
                              coords={'model': MODELS, 'basins': BASINS}),
        t4_mean=t4_mean,
        t4_sigma=t4_mean.copy())


def test_scaling_is_exact_and_proportional():
    """
    The ensemble is one run per ocean state scaled onto the parameter grid.
    That is exact because melt is proportional to the parameter, so the
    scaling must be too.
    """
    unit = xr.DataArray([3.0, 5.0], dims='basins')

    scaled = scale_to_ensemble(unit, PARAMETERS)

    for index, value in enumerate(PARAMETERS):
        np.testing.assert_allclose(
            scaled.isel(p1=index, p2=0).values, value * unit.values)


def test_scaling_adds_the_singleton_second_parameter():
    """
    The toolbox indexes terms by (p1, p2); the quadratic forms use only one
    parameter, so p2 is a singleton.
    """
    unit = xr.DataArray([1.0, 2.0], dims='basins')

    scaled = scale_to_ensemble(unit, PARAMETERS)

    assert scaled.dims[:2] == ('p1', 'p2')
    assert scaled.sizes['p2'] == 1
    assert scaled.sizes['p1'] == len(PARAMETERS)


def test_all_summands_are_weighted_by_default():
    """Passing None weights everything the ensemble provides."""
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS)

    assert (terms['t3_weights'] == 1).all()
    assert (terms['t4_weights'] == 1).all()


def test_restricting_j3_to_named_models():
    """A model left out of the weighting must not constrain the fit."""
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS,
                                t3_models=('mathiot',))

    weights = terms['t3_weights']
    assert (weights.sel(model='mathiot') == 1).all()
    assert (weights.sel(model='haid') == 0).all()


def test_restricting_j4_to_pig_zeroes_dotson():
    """
    The published weighting excludes Dotson entirely.  With one shelf J4
    constrains a single amplitude that either melt form can match by
    rescaling its parameter, so it stops discriminating between them --
    which is exactly why this is an option and not the default.
    """
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS,
                                t4_regions=('pig',))

    weights = terms['t4_weights']
    assert (weights.sel(region='pig') == 1).all()
    assert (weights.sel(region='dotson') == 0).all()


def test_restricting_j4_to_named_years():
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS,
                                t4_years=(2009, 2012))

    weights = terms['t4_weights']
    assert float(weights.sel(region='pig', year=2009)) == 1.0
    assert float(weights.sel(region='pig', year=2014)) == 0.0


def test_a_year_the_ensemble_did_not_run_carries_no_weight():
    """
    Asking for a year with no MALI run must not silently contribute; the
    weighting is intersected with what the ensemble actually provides.
    """
    units = _units()
    units['t4'] = units['t4'].sel(year=[2009, 2012])
    targets = _targets()

    terms = build_toolbox_terms(units, targets, PARAMETERS, t4_years=None)

    assert float(terms['t4_weights'].sel(region='pig', year=2014)) == 0.0


def test_published_and_full_weightings_differ():
    """
    The two weightings must not be accidentally identical, or the config
    option would be meaningless.
    """
    full = build_toolbox_terms(_units(), _targets(), PARAMETERS)
    published = build_toolbox_terms(_units(), _targets(), PARAMETERS,
                                    t3_models=('mathiot',),
                                    t4_regions=('pig',),
                                    t4_years=(2009, 2012))

    assert float(full['t4_weights'].sum()) > \
        float(published['t4_weights'].sum())
    assert float(full['t3_weights'].sum()) > \
        float(published['t3_weights'].sum())


def test_the_toolbox_receives_every_argument_it_needs():
    """A missing key would surface as an opaque TypeError at call time."""
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS)

    for term in ('t1', 't2', 't3', 't4'):
        for suffix in ('model', 'obs_mean', 'obs_sigma', 'weights'):
            assert f'{term}_{suffix}' in terms


def test_j3_targets_are_aligned_with_the_ensemble_models():
    """
    The targets carry every ocean model; selecting the ensemble's models
    keeps the two aligned rather than broadcasting against each other.
    """
    terms = build_toolbox_terms(_units(), _targets(), PARAMETERS)

    assert list(terms['t3_obs_mean'].model.values) == \
        list(terms['t3_model'].model.values)


def test_scaling_preserves_nan_for_empty_groups():
    """
    An aggregate that came out NaN because no cells contributed must stay
    NaN through the scaling, so it drops out of the objective.
    """
    unit = xr.DataArray([1.0, np.nan], dims='basins')

    scaled = scale_to_ensemble(unit, PARAMETERS)

    assert np.isnan(scaled.isel(p1=0, p2=0).values[1])


def test_scaling_by_zero_gives_zero_melt():
    """A zero parameter must give zero aggregate, not NaN."""
    unit = xr.DataArray([2.0], dims='basins')

    scaled = scale_to_ensemble(unit, np.array([0.0, 1.0]))

    assert float(scaled.isel(p1=0, p2=0)[0]) == pytest.approx(0.0)
