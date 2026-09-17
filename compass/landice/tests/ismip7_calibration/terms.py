"""
Objective-function terms J1-J4 for ISMIP7 melt-module calibration, on any mesh.

The reference implementation in
``parameterisations/parameter_selection_toolbox.py`` assumes a uniform
structured grid: it converts cell sums to Gt/yr with a single scalar
``cvt_m = reso**2 / 1e12`` and takes unweighted means over basins.  That is
correct for the ISMIP 8 km grid but wrong for a variable-resolution
unstructured mesh such as MALI's 4-20 km Antarctic mesh, where cell area
varies by a factor of ~25.

The functions here take an explicit cell-area array instead, and work over an
arbitrary set of cell dimensions.  Passing a uniform area reproduces the
structured-grid result exactly, so the same code serves both cases; see
``tests/test_terms.py``.

Conventions, matching the toolbox:

* melt rates are in kg m-2 yr-1, positive for melting, negative for refreezing
* integrated terms (J1, J2, J4) are in Gt yr-1
* averaged terms (J3) are in kg m-2 yr-1
* an aggregate that comes out exactly zero is set to NaN, so that regions
  with no contributing cells drop out of the objective function rather than
  being treated as a real zero
"""

import numpy as np
import xarray as xr

#: kg per Gt
KG_PER_GT = 1.0e12

#: name of the flattened cell dimension used internally
CELL_DIM = '_cell'


def stack_cells(da, cell_dims):
    """
    Flatten the cell dimensions of ``da`` into a single dimension.

    Parameters
    ----------
    da : xarray.DataArray
        Array with one or more cell dimensions.
    cell_dims : sequence of str
        The dimensions to flatten, e.g. ``('y', 'x')`` for a structured grid or
        ``('nCells',)`` for an MPAS mesh.

    Returns
    -------
    xarray.DataArray
        ``da`` with ``cell_dims`` replaced by :data:`CELL_DIM`.
    """
    cell_dims = tuple(cell_dims)
    missing = [dim for dim in cell_dims if dim not in da.dims]
    if missing:
        raise ValueError(f'{missing} not found in dimensions {tuple(da.dims)}')
    if len(cell_dims) == 1 and cell_dims[0] == CELL_DIM:
        return da
    stacked = da.stack({CELL_DIM: cell_dims})
    # drop the MultiIndex coordinate; we only need positional alignment and the
    # index would otherwise have to be carried through every groupby
    return stacked.drop_vars(
        [CELL_DIM, *cell_dims], errors='ignore'
    ).assign_coords({CELL_DIM: np.arange(stacked.sizes[CELL_DIM])})


def _prepare(melt, area, mask, groups, cell_dims):
    """Flatten melt, area, mask and groups onto a common cell dimension."""
    melt = stack_cells(melt, cell_dims)
    area = stack_cells(area, cell_dims)
    groups = stack_cells(groups, cell_dims)
    if mask is None:
        valid = xr.ones_like(area, dtype=bool)
    else:
        valid = stack_cells(mask, cell_dims).astype(bool)
    # cells with no group assignment never contribute
    if groups.dtype.kind == 'f':
        valid = valid & groups.notnull()
    return melt, area, valid, groups


def integrate_by_group(
    melt, area, mask, groups, cell_dims, group_dim='basins'
):
    """
    Integrate a melt rate over each group, in Gt yr-1.

    This is the mesh-agnostic form of the toolbox's
    ``melt.where(mask).groupby(groups).sum() * reso**2 / 1e12``.

    Parameters
    ----------
    melt : xarray.DataArray
        Melt rate in kg m-2 yr-1.  May carry any number of extra dimensions
        (parameter values, ocean model, year, ...); they are preserved.
    area : xarray.DataArray
        Cell area in m2, over ``cell_dims`` only.
    mask : xarray.DataArray or None
        Boolean; True where the cell should contribute (e.g. floating ice).
    groups : xarray.DataArray
        Group id per cell (basin number, buttressing bin, ...).
    cell_dims : sequence of str
        Dimensions of ``melt`` that index cells.
    group_dim : str, optional
        Name to give the resulting group dimension.

    Returns
    -------
    xarray.DataArray
        Integrated melt in Gt yr-1, with ``cell_dims`` replaced by
        ``group_dim``.  Groups whose total is exactly zero are set to NaN.
    """
    melt, area, valid, groups = _prepare(melt, area, mask, groups, cell_dims)

    weighted = (melt * area).where(valid)
    total = weighted.groupby(groups.rename(group_dim)).sum(skipna=True)
    total = total / KG_PER_GT
    return total.where(total != 0.0)


def average_by_group(melt, area, mask, groups, cell_dims, group_dim='basins'):
    """
    Area-weighted mean melt rate over each group, in kg m-2 yr-1.

    This is the mesh-agnostic form of the toolbox's
    ``melt.where(mask).groupby(groups).mean()``.  On a uniform grid an
    area-weighted mean and a plain mean coincide; on a variable-resolution mesh
    they do not, and the area-weighted one is the physically meaningful choice.

    Parameters and returns are as for :func:`integrate_by_group`, except
    that the result is a mean rather than an integral.
    """
    melt, area, valid, groups = _prepare(melt, area, mask, groups, cell_dims)

    grouper = groups.rename(group_dim)
    weighted = (melt * area).where(valid)
    # only count area where melt is defined, so that NaN melt does not bias the
    # denominator
    weights = area.where(valid & melt.notnull())

    numer = weighted.groupby(grouper).sum(skipna=True)
    denom = weights.groupby(grouper).sum(skipna=True)
    mean = numer / denom.where(denom != 0.0)
    return mean.where(mean != 0.0)


def calculate_term1(
    ensemble, area, mask, basins, melt_obs, cell_dims, var='melt_rate'
):
    """
    J1: basin-integrated present-day melt, in Gt yr-1.

    Parameters
    ----------
    ensemble : xarray.Dataset
        Present-day ensemble, with ``var`` indexed by at least ``p1`` and
        ``p2``.
    area, mask, basins : xarray.DataArray
        Cell area (m2), contributing-cell mask, and IMBIE2 basin number per
        cell.  Basin numbers follow the **ISMIP7 0-based convention**,
        which differs by one from MALI's; see the basin-numbering note in
        the developer guide for why that matters.
    melt_obs : pandas.DataFrame
        Observational targets, with columns ``'BMR (Gt/yr)'`` and
        ``'BMR uncert (Gt/yr)'``, one row per basin in basin order.
    cell_dims : sequence of str
        Cell dimensions of the ensemble.
    var : str, optional
        Name of the melt-rate variable.

    Returns
    -------
    model, obs_mean, obs_sigma : xarray.DataArray
    """
    model = integrate_by_group(
        ensemble[var], area, mask, basins, cell_dims, group_dim='basins'
    )
    obs_mean = xr.DataArray(
        data=np.asarray(melt_obs['BMR (Gt/yr)'].values, dtype=float),
        name='melt_Gt_per_y',
        dims=['basin'],
        coords={'basin': np.arange(len(melt_obs))},
    )
    obs_sigma = xr.DataArray(
        data=np.asarray(melt_obs['BMR uncert (Gt/yr)'].values, dtype=float),
        name='melt_unc_Gt_per_y',
        dims=['basin'],
        coords={'basin': np.arange(len(melt_obs))},
    )
    return model, obs_mean, obs_sigma


def calculate_term2(
    ensemble, area, mask, bfrn_bins, target, cell_dims, var='melt_rate'
):
    """
    J2: melt integrated over buttressing (BFRN) bins, in Gt yr-1.

    ``target`` is the dataset holding ``melt_mean`` and ``melt_mean_err`` per
    bin.  Other parameters are as for :func:`calculate_term1`.
    """
    model = integrate_by_group(
        ensemble[var], area, mask, bfrn_bins, cell_dims, group_dim='BFRN_bins'
    )
    return model, target['melt_mean'], target['melt_mean_err']


def calculate_term3(
    cold_ensemble,
    warm_ensemble,
    cold_target,
    warm_target,
    area,
    mask,
    basins,
    cell_dims,
    var='melt_rate',
):
    """
    J3: warm-minus-cold basin-mean melt difference, in kg m-2 yr-1.

    The model term is the difference of area-weighted basin means between the
    warm and cold ocean states; the target is the corresponding difference from
    the ocean models, with uncertainties combined in quadrature.
    """
    cold = average_by_group(
        cold_ensemble[var], area, mask, basins, cell_dims, group_dim='basins'
    )
    warm = average_by_group(
        warm_ensemble[var], area, mask, basins, cell_dims, group_dim='basins'
    )
    model = warm - cold

    obs_mean = warm_target.melt_rate - cold_target.melt_rate
    obs_sigma = np.sqrt(
        warm_target.melt_rate_uncert**2 + cold_target.melt_rate_uncert**2
    )
    return model, obs_mean, obs_sigma


def calculate_term4(
    obs_ensemble,
    area,
    mask,
    region_label,
    target,
    cell_dims,
    var='melt_rate',
):
    """
    J4: PIG/Dotson integrated melt per observation year, in Gt yr-1.

    ``region_label`` labels each cell with an ice-shelf name (``'pig'``,
    ``'dotson'``); ``target`` holds ``melt_rate`` and ``melt_rate_uncert``
    indexed by region and year.  The result is reindexed onto the target so the
    region ordering matches.
    """
    model = integrate_by_group(
        obs_ensemble[var],
        area,
        mask,
        region_label,
        cell_dims,
        group_dim='region',
    )
    model = model.where(target.melt_rate.notnull())
    model = model.reindex_like(target.melt_rate)
    return model, target.melt_rate, target.melt_rate_uncert


def uniform_area(template, resolution, cell_dims):
    """
    Build a uniform cell-area array, for structured grids.

    Convenience so that a structured-grid case can call the same functions:
    ``area = uniform_area(melt, 8000.0, ('y', 'x'))``.

    Parameters
    ----------
    template : xarray.DataArray
        Any array carrying the cell dimensions and their coordinates.
    resolution : float
        Grid spacing in m; cell area is ``resolution**2``.
    cell_dims : sequence of str
        The cell dimensions.

    Returns
    -------
    xarray.DataArray
        Constant array of ``resolution**2``, over ``cell_dims`` only.
    """
    coords = {
        dim: template.coords[dim]
        for dim in cell_dims
        if dim in template.coords
    }
    shape = tuple(template.sizes[dim] for dim in cell_dims)
    return xr.DataArray(
        np.full(shape, float(resolution) ** 2),
        dims=tuple(cell_dims),
        coords=coords,
        name='area',
    )
