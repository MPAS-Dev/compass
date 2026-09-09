"""
Driving the ISMIP7 objective function.

The vendored toolbox
(:py:mod:`compass.landice.tests.ismip7_calibration.toolbox`) implements
protocol Eq. (3): it draws random term weights and random targets within
their uncertainties, and for each draw picks the parameter value that
minimises ``I = sum_i a_i J_i / median(J_i)``.  The distribution of those
minimisers gives the 5th / 50th / 95th percentiles the ISMIP7 projections
need.

This module assembles the arguments the toolbox expects and reduces its
output to those percentiles.  It is shared by the ``replication`` test case,
which drives it with melt from the Python reference implementation on the
ISMIP 8 km grid, and by the ``ais`` test case, which drives it with melt
from MALI on a MALI mesh.  Keeping one implementation means the MALI
calibration is validated by the replication reproducing published numbers.

**Melt is exactly linear in the calibration parameter**, so each term is
built once at a unit parameter value and scaled onto the parameter grid.
That is exact, not an approximation, and it is why the ensemble needs one
model run per ocean state rather than one per (state, parameter) pair.
"""

import numpy as np
import xarray as xr

from compass.landice.tests.ismip7_calibration.toolbox import (
    parameter_selection_toolbox as toolbox,
)


def scale_to_ensemble(unit_aggregate, param_values):
    """
    Turn a unit-parameter aggregate into the ``(p1, p2, ...)`` ensemble.

    The toolbox wants each modelled term indexed by ``(p1, p2, ...)``, where
    ``p1`` is the melt parameter being selected and ``p2`` a second parameter
    that the quadratic parameterizations do not use, so it is a singleton.

    Parameters
    ----------
    unit_aggregate : xarray.DataArray
        An aggregate (J1, J2, J3 or J4) computed at a parameter value of 1

    param_values : numpy.ndarray
        The parameter grid to scale onto

    Returns
    -------
    scaled : xarray.DataArray
        ``unit_aggregate`` broadcast over ``p1`` and a singleton ``p2``
    """
    p1 = xr.DataArray(param_values, dims=['p1'],
                      coords={'p1': param_values})
    out = (unit_aggregate * p1).expand_dims({'p2': np.ones(1)})
    return out.transpose('p1', 'p2', ...)


def build_toolbox_terms(units, targets, param_values, t3_models=None,
                        t4_regions=None, t4_years=None):
    """
    Assemble the arguments ``calculate_objective_function`` expects.

    Parameters
    ----------
    units : dict
        Unit-parameter aggregates ``t1``, ``t2``, ``t3`` and ``t4``, as
        produced by :py:mod:`compass.landice.tests.ismip7_calibration.terms`

    targets : dict
        Observational targets, from
        :py:func:`compass.landice.tests.ismip7_calibration.datasets.load_targets`

    param_values : numpy.ndarray
        The parameter grid

    t3_models : sequence of str, optional
        Ocean models to give non-zero weight in J3.  None weights every model
        the ensemble provides.

    t4_regions : sequence of str, optional
        Ice shelves to give non-zero weight in J4, from ``'pig'`` and
        ``'dotson'``.  None weights both.

        The published weighting uses PIG alone, which is 2 of the 18
        available observations.  As weighted that way, J4 constrains a single
        amplitude that either melt form can match by rescaling its parameter,
        so it cannot discriminate between forms; including Dotson makes it a
        relative constraint between two shelves, which it can.

    t4_years : sequence of int, optional
        Observation years to give non-zero weight in J4.  None weights every
        year the ensemble provides.

    Returns
    -------
    terms : dict
        Keyword arguments for
        ``toolbox.calculate_objective_function``
    """  # noqa: E501
    t1_model = scale_to_ensemble(units['t1'], param_values)
    t2_model = scale_to_ensemble(units['t2'], param_values)
    t3_model = scale_to_ensemble(units['t3'], param_values)

    t4_model = scale_to_ensemble(units['t4'], param_values)
    t4_model = t4_model.where(targets['t4_mean'].notnull())
    t4_model = t4_model.reindex_like(targets['t4_mean'])

    t1_weights = xr.DataArray(
        np.ones(t1_model.sizes['basins']), dims=['basins'],
        coords={'basins': t1_model.basins.values})

    t3_weights = xr.DataArray(
        np.ones((t3_model.sizes['model'], t3_model.sizes['basins'])),
        dims=['model', 'basins'],
        coords={'model': t3_model.model.values,
                'basins': t3_model.basins.values})
    if t3_models is not None:
        t3_weights = t3_weights.where(
            t3_weights.model.isin(list(t3_models)), other=0)

    t4_weights = xr.DataArray(
        np.ones((targets['t4_mean'].sizes['region'],
                 targets['t4_mean'].sizes['year'])),
        dims=['region', 'year'],
        coords={'region': targets['t4_mean'].region.values,
                'year': targets['t4_mean'].year.values})
    if t4_regions is not None:
        t4_weights = t4_weights.where(
            t4_weights.region.isin(list(t4_regions)), other=0)
    if t4_years is not None:
        t4_weights = t4_weights.where(
            t4_weights.year.isin(list(t4_years)), other=0)
    # a year the ensemble did not run cannot contribute, whatever the
    # weighting asks for
    t4_weights = t4_weights.where(
        t4_weights.year.isin(list(units['t4'].year.values)), other=0)

    return dict(
        t1_model=t1_model,
        t1_obs_mean=targets['t1_mean'],
        t1_obs_sigma=targets['t1_sigma'],
        t1_weights=t1_weights,
        t2_model=t2_model,
        t2_obs_mean=targets['t2_mean'],
        t2_obs_sigma=targets['t2_sigma'],
        t2_weights=targets['t2_weights'],
        t3_model=t3_model,
        t3_obs_mean=targets['t3_mean'].sel(model=t3_model.model.values),
        t3_obs_sigma=targets['t3_sigma'].sel(model=t3_model.model.values),
        t3_weights=t3_weights,
        t4_model=t4_model,
        t4_obs_mean=targets['t4_mean'],
        t4_obs_sigma=targets['t4_sigma'],
        t4_weights=t4_weights)


def run_optimisation(terms, param_values, resolution=8000.0,
                     sample_size=100000, seed=None):
    """
    Sample the objective function and return the parameter distribution.

    Parameters
    ----------
    terms : dict
        From :py:func:`build_toolbox_terms`

    param_values : numpy.ndarray
        The parameter grid the terms were built on

    resolution : float, optional
        Passed through to the toolbox; it does not use it for these terms,
        which arrive already aggregated

    sample_size : int, optional
        Number of random draws of the term weights and the targets

    seed : int, optional
        Seed for the random draws.  Set it so that the reported percentiles
        are reproducible; the toolbox draws from the global numpy random
        state.

    Returns
    -------
    result : dict
        ``p5``, ``median``, ``p95``, ``mode`` and the raw ``min_p1``
    """
    if seed is not None:
        np.random.seed(seed)

    min_p1, _ = toolbox.calculate_objective_function(
        sample_size,
        resolution,
        terms['t1_model'], terms['t1_obs_mean'], terms['t1_obs_sigma'],
        terms['t1_weights'],
        terms['t2_model'], terms['t2_obs_mean'], terms['t2_obs_sigma'],
        terms['t2_weights'],
        terms['t3_model'], terms['t3_obs_mean'], terms['t3_obs_sigma'],
        terms['t3_weights'],
        terms['t4_model'], terms['t4_obs_mean'], terms['t4_obs_sigma'],
        terms['t4_weights'])
    min_p1 = np.asarray(min_p1, dtype=float)

    values = np.asarray(param_values, dtype=float)
    step = np.diff(values).min()
    edges = np.append(values[0] - 0.5 * step, values + 1.0e-7 * step)
    counts, _ = np.histogram(min_p1, bins=edges)

    return dict(min_p1=min_p1,
                p5=float(np.percentile(min_p1, 5)),
                median=float(np.median(min_p1)),
                p95=float(np.percentile(min_p1, 95)),
                mode=float(values[int(np.argmax(counts))]))
