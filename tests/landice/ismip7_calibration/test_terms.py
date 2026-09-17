"""
Tests for the area-weighted, mesh-agnostic objective-function terms.

The point of these functions is that they give the *same* answer as the
upstream toolbox on a uniform grid, while being correct on a
variable-resolution mesh where the toolbox's single scalar cell area is not.
Both halves of that claim are tested here.
"""

import numpy as np
import pytest
import xarray as xr

from compass.landice.tests.ismip7_calibration.terms import (
    KG_PER_GT,
    average_by_group,
    integrate_by_group,
    stack_cells,
    uniform_area,
)


def _structured(melt, groups, resolution=8000.0):
    """Build a small structured-grid case."""
    ny, nx = melt.shape
    coords = {'y': np.arange(ny, dtype=float),
              'x': np.arange(nx, dtype=float)}
    melt_da = xr.DataArray(melt, dims=('y', 'x'), coords=coords)
    groups_da = xr.DataArray(groups, dims=('y', 'x'), coords=coords)
    area = uniform_area(melt_da, resolution, ('y', 'x'))
    mask = xr.ones_like(melt_da, dtype=bool)
    return melt_da, area, mask, groups_da


def test_uniform_area_matches_the_toolbox_convention():
    """
    On a uniform grid the integral must equal the toolbox's
    ``sum * reso**2 / 1e12``.
    """
    melt = np.array([[1.0, 2.0], [3.0, 4.0]])
    groups = np.array([[0, 0], [1, 1]])
    resolution = 8000.0
    melt_da, area, mask, groups_da = _structured(melt, groups, resolution)

    result = integrate_by_group(melt_da, area, mask, groups_da, ('y', 'x'))

    expected_0 = (1.0 + 2.0) * resolution**2 / KG_PER_GT
    expected_1 = (3.0 + 4.0) * resolution**2 / KG_PER_GT
    assert float(result.sel(basins=0)) == pytest.approx(expected_0)
    assert float(result.sel(basins=1)) == pytest.approx(expected_1)


def test_integral_is_area_weighted_on_a_variable_mesh():
    """
    With unequal cell areas the integral must weight by area.  A plain sum
    would give the wrong answer, which is the bug these functions exist to
    avoid on a 4-20 km mesh.
    """
    melt = xr.DataArray([1.0, 1.0, 1.0], dims='nCells')
    area = xr.DataArray([1.0e9, 2.0e9, 4.0e9], dims='nCells')
    groups = xr.DataArray([0, 0, 0], dims='nCells')
    mask = xr.ones_like(melt, dtype=bool)

    result = integrate_by_group(melt, area, mask, groups, ('nCells',))

    assert float(result.sel(basins=0)) == pytest.approx(7.0e9 / KG_PER_GT)


def test_mean_is_area_weighted_not_a_plain_mean():
    """A big cold cell must outweigh a small warm one."""
    melt = xr.DataArray([0.0, 100.0], dims='nCells')
    area = xr.DataArray([9.0, 1.0], dims='nCells')
    groups = xr.DataArray([0, 0], dims='nCells')
    mask = xr.ones_like(melt, dtype=bool)

    result = average_by_group(melt, area, mask, groups, ('nCells',))

    # area-weighted mean is 10, the plain mean would be 50
    assert float(result.sel(basins=0)) == pytest.approx(10.0)


def test_mask_excludes_cells_from_both_sums():
    """
    A masked cell must not contribute to the numerator or the denominator
    of an area-weighted mean.
    """
    melt = xr.DataArray([10.0, 1000.0], dims='nCells')
    area = xr.DataArray([1.0, 1.0], dims='nCells')
    groups = xr.DataArray([0, 0], dims='nCells')
    mask = xr.DataArray([True, False], dims='nCells')

    result = average_by_group(melt, area, mask, groups, ('nCells',))

    assert float(result.sel(basins=0)) == pytest.approx(10.0)


def test_nan_melt_does_not_bias_the_mean_denominator():
    """
    Where melt is NaN the cell's area must be left out of the weights, or
    the mean is diluted toward zero.
    """
    melt = xr.DataArray([10.0, np.nan], dims='nCells')
    area = xr.DataArray([1.0, 99.0], dims='nCells')
    groups = xr.DataArray([0, 0], dims='nCells')
    mask = xr.ones_like(melt, dtype=bool)

    result = average_by_group(melt, area, mask, groups, ('nCells',))

    assert float(result.sel(basins=0)) == pytest.approx(10.0)


def test_groups_are_kept_separate():
    """Cells in different groups must not be mixed."""
    melt = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims='nCells')
    area = xr.DataArray([1.0e12, 1.0e12, 1.0e12, 1.0e12], dims='nCells')
    groups = xr.DataArray([0, 1, 0, 1], dims='nCells')
    mask = xr.ones_like(melt, dtype=bool)

    result = integrate_by_group(melt, area, mask, groups, ('nCells',))

    assert float(result.sel(basins=0)) == pytest.approx(4.0)
    assert float(result.sel(basins=1)) == pytest.approx(6.0)


def test_empty_group_becomes_nan_not_zero():
    """
    A group with no contributing cells must drop out of the objective
    rather than be treated as a real zero.
    """
    melt = xr.DataArray([1.0, 2.0], dims='nCells')
    area = xr.DataArray([1.0e12, 1.0e12], dims='nCells')
    groups = xr.DataArray([0, 1], dims='nCells')
    mask = xr.DataArray([True, False], dims='nCells')

    result = integrate_by_group(melt, area, mask, groups, ('nCells',))

    assert np.isnan(float(result.sel(basins=1)))


def test_extra_dimensions_are_preserved():
    """
    Terms carry parameter, model and year dimensions through the
    aggregation untouched.
    """
    melt = xr.DataArray(np.ones((3, 4)), dims=('p1', 'nCells'))
    area = xr.DataArray(np.full(4, 1.0e12), dims='nCells')
    groups = xr.DataArray([0, 0, 1, 1], dims='nCells')
    mask = xr.ones_like(area, dtype=bool)

    result = integrate_by_group(melt, area, mask, groups, ('nCells',))

    assert result.sizes['p1'] == 3
    assert result.sizes['basins'] == 2


def test_aggregation_is_linear_in_melt():
    """
    Melt is proportional to the calibration parameter, and the calibration
    relies on the aggregates being proportional to it too.
    """
    melt = xr.DataArray([1.0, 2.0, 3.0], dims='nCells')
    area = xr.DataArray([1.0e12, 2.0e12, 3.0e12], dims='nCells')
    groups = xr.DataArray([0, 0, 0], dims='nCells')
    mask = xr.ones_like(melt, dtype=bool)

    once = integrate_by_group(melt, area, mask, groups, ('nCells',))
    twice = integrate_by_group(2.0 * melt, area, mask, groups, ('nCells',))

    assert float(twice.sel(basins=0)) == pytest.approx(
        2.0 * float(once.sel(basins=0)))


def test_stack_cells_flattens_structured_dimensions():
    """A (y, x) field is flattened onto one cell dimension."""
    da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=('y', 'x'))

    stacked = stack_cells(da, ('y', 'x'))

    assert stacked.sizes == {'_cell': 6}
    assert np.array_equal(np.sort(stacked.values), np.arange(6.0))


def test_stack_cells_rejects_a_missing_dimension():
    """A typo in the cell dimensions should fail loudly."""
    da = xr.DataArray(np.zeros(3), dims='nCells')

    with pytest.raises(ValueError, match='not found in dimensions'):
        stack_cells(da, ('ncells',))
