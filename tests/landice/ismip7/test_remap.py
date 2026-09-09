"""
Tests for the shared ISMIP7 remapping helpers.
"""

import logging

import numpy as np
import xarray as xr

from compass.landice.ismip7.remap import extrapolate_source


def _logger():
    logger = logging.getLogger('test_remap')
    logger.addHandler(logging.NullHandler())
    return logger


def _write_source(path, values, time_units=None):
    """Write a small (time, y, x) source file."""
    ds = xr.Dataset()
    ds['field'] = (('time', 'y', 'x'), values)
    ds['time'] = ('time', np.arange(values.shape[0]))
    if time_units is not None:
        ds['time'].attrs['units'] = time_units
    ds.to_netcdf(path)
    ds.close()


def test_extrapolate_fills_all_gaps(tmp_path):
    """Every missing value should be replaced by a valid neighbour."""
    values = np.arange(18.0).reshape(2, 3, 3)
    values[0, 1, 1] = np.nan
    values[1, 0, 0] = np.nan
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    _write_source(source, values)

    extrapolate_source(str(source), str(output), 'field', _logger())

    with xr.open_dataset(output) as ds:
        filled = ds['field'].values
    assert np.isfinite(filled).all()


def test_extrapolate_takes_the_nearest_value(tmp_path):
    """A gap is filled from its nearest valid cell, not by interpolation."""
    values = np.full((1, 1, 5), np.nan)
    values[0, 0, 0] = 10.0
    values[0, 0, 4] = 20.0
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    _write_source(source, values)

    extrapolate_source(str(source), str(output), 'field', _logger())

    with xr.open_dataset(output) as ds:
        filled = ds['field'].values[0, 0]
    # nearest-neighbour fill, so every cell takes one of the two valid values
    assert set(np.unique(filled)) <= {10.0, 20.0}
    assert filled[1] == 10.0
    assert filled[3] == 20.0


def test_extrapolate_leaves_valid_data_untouched(tmp_path):
    """Cells that were already valid must not change."""
    values = np.arange(9.0).reshape(1, 3, 3)
    values[0, 1, 1] = np.nan
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    _write_source(source, values)

    extrapolate_source(str(source), str(output), 'field', _logger())

    with xr.open_dataset(output) as ds:
        filled = ds['field'].values
    valid = np.isfinite(values)
    assert np.array_equal(filled[valid], values[valid])


def test_extrapolate_accepts_several_variables(tmp_path):
    """A list of variable names extrapolates each of them."""
    ds = xr.Dataset()
    first = np.array([[[1.0, np.nan]]])
    second = np.array([[[np.nan, 2.0]]])
    ds['first'] = (('time', 'y', 'x'), first)
    ds['second'] = (('time', 'y', 'x'), second)
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    ds.to_netcdf(source)
    ds.close()

    extrapolate_source(str(source), str(output), ['first', 'second'],
                       _logger())

    with xr.open_dataset(output) as result:
        assert np.isfinite(result['first'].values).all()
        assert np.isfinite(result['second'].values).all()


def test_extrapolate_handles_non_cf_time(tmp_path):
    """
    The fracture forcing carries units="year", which xarray cannot decode.

    Those callers pass decode_times=False; the default stays True so that
    the ocean and atmosphere callers keep their previous behaviour.
    """
    values = np.array([[[1.0, np.nan]]])
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    _write_source(source, values, time_units='year')

    extrapolate_source(str(source), str(output), 'field', _logger(),
                       decode_times=False)

    with xr.open_dataset(output, decode_times=False) as ds:
        assert np.isfinite(ds['field'].values).all()
        assert ds['time'].attrs['units'] == 'year'


def test_extrapolate_is_a_no_op_without_gaps(tmp_path):
    """Data with no missing values passes through unchanged."""
    values = np.arange(8.0).reshape(2, 2, 2)
    source = tmp_path / 'source.nc'
    output = tmp_path / 'output.nc'
    _write_source(source, values)

    extrapolate_source(str(source), str(output), 'field', _logger())

    with xr.open_dataset(output) as ds:
        assert np.array_equal(ds['field'].values, values)
