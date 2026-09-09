"""
Tests for recomputing MALI's melt from its diagnostic output.

The vertical interpolation here is written from the protocol rather than
transliterated from MALI's Fortran, so that comparing the two in the
``verify_melt`` step is a real check.  These tests pin its four code paths
and the area-weighted basin mean.
"""

import numpy as np
import pytest
import xarray as xr

from compass.landice.tests.ismip7_calibration.ais.melt_model import (
    FREEZING_TEMP_DEPTH_DEPENDENCE,
    basin_mean_tf,
    initial_draft,
    integrate_by_basin,
    interpolate_to_draft,
)
from compass.landice.tests.ismip7_calibration.quadratic import MALI

#: three layer centres, negative downward as MALI expects
Z_OCEAN = np.array([-30.0, -90.0, -150.0])


def _field(values):
    """One cell with a value per layer."""
    return np.array([values], dtype=float)


def test_interpolation_above_the_shallowest_centre():
    """A draft above the top layer centre takes the top layer's value."""
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([-10.0]), np.array([-500.0]))

    assert result[0] == pytest.approx(1.0)


def test_interpolation_below_the_deepest_centre():
    """A draft below the bottom layer centre takes the bottom value."""
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([-300.0]), np.array([-500.0]))

    assert result[0] == pytest.approx(3.0)


def test_interpolation_when_the_layer_below_is_beneath_the_bed():
    """
    Where the next layer down is below the bed there is no water there, so
    the shallower layer is used rather than interpolating into rock.
    """
    # the draft is at -60 and the bed at -80, so the -90 layer centre is
    # below the bed and there is no water there to interpolate into
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([-60.0]), np.array([-80.0]))

    assert result[0] == pytest.approx(1.0)


def test_interpolation_between_layer_centres():
    """Half way between two centres gives the average of their values."""
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([-60.0]), np.array([-500.0]))

    assert result[0] == pytest.approx(1.5)


def test_interpolation_is_linear_in_depth():
    """A quarter of the way down gives a quarter of the difference."""
    result = interpolate_to_draft(_field([0.0, 4.0, 8.0]), Z_OCEAN,
                                  np.array([-45.0]), np.array([-500.0]))

    assert result[0] == pytest.approx(1.0)


def test_interpolation_reproduces_layer_centre_values_exactly():
    """At a layer centre the result must be that layer's value."""
    values = [1.0, 2.0, 3.0]
    for index, depth in enumerate(Z_OCEAN):
        result = interpolate_to_draft(_field(values), Z_OCEAN,
                                      np.array([depth]),
                                      np.array([-500.0]))
        assert result[0] == pytest.approx(values[index])


def test_interpolation_handles_many_cells_independently():
    """Each cell uses its own draft and bed."""
    field = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    result = interpolate_to_draft(field, Z_OCEAN,
                                  np.array([-10.0, -300.0]),
                                  np.array([-500.0, -500.0]))

    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(3.0)


def test_freezing_correction_applies_below_the_deepest_centre():
    """
    MALI corrects the thermal forcing for the depth dependence of the
    freezing point where it extrapolates downward from the deepest layer.
    """
    draft = -300.0
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([draft]), np.array([-500.0]),
                                  freezing_correction=True)

    expected = 3.0 - (Z_OCEAN[-1] - draft) * FREEZING_TEMP_DEPTH_DEPENDENCE
    assert result[0] == pytest.approx(expected)


def test_freezing_correction_applies_below_the_bed():
    """
    The same correction applies in the other downward-extrapolating branch,
    where the layer below the draft is beneath the bed.  Omitting it here
    was a real bug the verification step caught.
    """
    draft = -60.0
    result = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                  np.array([draft]), np.array([-80.0]),
                                  freezing_correction=True)

    expected = 1.0 - (Z_OCEAN[0] - draft) * FREEZING_TEMP_DEPTH_DEPENDENCE
    assert result[0] == pytest.approx(expected)


def test_freezing_correction_does_not_apply_when_interpolating():
    """
    Between two layer centres there is no extrapolation, so the correction
    must not be applied.
    """
    plain = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                 np.array([-60.0]), np.array([-500.0]))
    corrected = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                     np.array([-60.0]), np.array([-500.0]),
                                     freezing_correction=True)

    assert plain[0] == pytest.approx(corrected[0])


def test_freezing_correction_does_not_apply_above_the_shallowest_centre():
    """The draft is above the layer, so there is nothing to correct."""
    plain = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                 np.array([-10.0]), np.array([-500.0]))
    corrected = interpolate_to_draft(_field([1.0, 2.0, 3.0]), Z_OCEAN,
                                     np.array([-10.0]), np.array([-500.0]),
                                     freezing_correction=True)

    assert plain[0] == pytest.approx(corrected[0])


def test_basin_mean_is_area_weighted():
    """
    MALI's non-local form uses an area-weighted basin mean; a plain mean
    would be wrong on a variable-resolution mesh.
    """
    tf = xr.DataArray([0.0, 10.0], dims='nCells')
    area = xr.DataArray([9.0, 1.0], dims='nCells')
    floating = xr.DataArray([True, True], dims='nCells')
    basin = xr.DataArray([1, 1], dims='nCells')

    result = basin_mean_tf(tf, area, floating, basin)

    np.testing.assert_allclose(result.values, [1.0, 1.0])


def test_basin_mean_excludes_grounded_cells():
    """Only floating cells contribute to the mean."""
    tf = xr.DataArray([100.0, 2.0], dims='nCells')
    area = xr.DataArray([1.0, 1.0], dims='nCells')
    floating = xr.DataArray([False, True], dims='nCells')
    basin = xr.DataArray([1, 1], dims='nCells')

    result = basin_mean_tf(tf, area, floating, basin)

    np.testing.assert_allclose(result.values, [2.0, 2.0])


def test_basin_mean_keeps_basins_separate():
    """Each cell gets the mean of its own basin."""
    tf = xr.DataArray([1.0, 3.0, 10.0], dims='nCells')
    area = xr.DataArray([1.0, 1.0, 1.0], dims='nCells')
    floating = xr.DataArray([True, True, True], dims='nCells')
    basin = xr.DataArray([1, 1, 2], dims='nCells')

    result = basin_mean_tf(tf, area, floating, basin)

    np.testing.assert_allclose(result.values, [2.0, 2.0, 10.0])


def test_integrate_by_basin_converts_to_gigatonnes():
    """kg m-2 yr-1 times m2 is kg yr-1; the result must be Gt yr-1."""
    melt = xr.DataArray([1.0], dims='nCells')
    area = xr.DataArray([1.0e12], dims='nCells')
    floating = xr.DataArray([True], dims='nCells')
    basin = xr.DataArray([3], dims='nCells')

    result = integrate_by_basin(melt, area, floating, basin)

    assert float(result.sel(basin=3)) == pytest.approx(1.0)


def test_initial_draft_uses_flotation_where_floating(tmp_path):
    """
    The draft must be reconstructed from the mesh, not from the run output,
    because melt thins the ice over the timestep.
    """
    path = tmp_path / 'mesh.nc'
    ds = xr.Dataset()
    ds['thickness'] = ('nCells', np.array([1000.0]))
    ds['bedTopography'] = ('nCells', np.array([-2000.0]))
    ds.to_netcdf(path)
    ds.close()

    draft = initial_draft(str(path))

    expected = -MALI.rho_ice / MALI.rho_ocean * 1000.0
    assert float(draft[0]) == pytest.approx(expected)


def test_initial_draft_uses_the_bed_where_grounded(tmp_path):
    """Grounded ice sits on the bed, above its flotation draft."""
    path = tmp_path / 'mesh.nc'
    ds = xr.Dataset()
    ds['thickness'] = ('nCells', np.array([100.0]))
    ds['bedTopography'] = ('nCells', np.array([-50.0]))
    ds.to_netcdf(path)
    ds.close()

    draft = initial_draft(str(path))

    assert float(draft[0]) == pytest.approx(-50.0)


def test_initial_draft_handles_a_time_dimension(tmp_path):
    """MALI mesh files may carry a singleton Time dimension."""
    path = tmp_path / 'mesh.nc'
    ds = xr.Dataset()
    ds['thickness'] = (('Time', 'nCells'), np.array([[1000.0]]))
    ds['bedTopography'] = (('Time', 'nCells'), np.array([[-2000.0]]))
    ds.to_netcdf(path)
    ds.close()

    draft = initial_draft(str(path))

    assert draft.dims == ('nCells',)
