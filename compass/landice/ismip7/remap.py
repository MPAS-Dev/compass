"""
Shared helpers for remapping ISMIP7 forcing data onto the MALI mesh.
"""
import os

import netCDF4
import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt


def extrapolate_source(input_file, output_file, varnames, logger):
    """
    Extrapolate fill/missing values on the source polar stereographic grid
    using nearest-neighbor via ``distance_transform_edt``. This must be done
    before remapping so that fill values don't contaminate the interpolation
    stencil.

    Parameters
    ----------
    input_file : str
        Path to the input NetCDF file on the source grid

    output_file : str
        Path to write the extrapolated file

    varnames : str or list of str
        Name(s) of the variable(s) to extrapolate

    logger : logging.Logger
        Logger for status messages
    """
    if isinstance(varnames, str):
        varnames = [varnames]

    logger.info(f"    Extrapolating fill values on source grid: "
                f"{os.path.basename(input_file)}")

    ds = xr.open_dataset(input_file, decode_times=False)

    for varname in varnames:
        data = ds[varname]
        values = data.values.copy()
        non_spatial_shape = values.shape[:-2]

        for idx in np.ndindex(non_spatial_shape):
            slab = values[idx]
            valid_mask = np.isfinite(slab)
            if valid_mask.all() or not valid_mask.any():
                continue
            nearest_inds = distance_transform_edt(
                ~valid_mask, return_distances=False, return_indices=True)
            invalid = ~valid_mask
            values[idx][invalid] = slab[
                nearest_inds[0, invalid],
                nearest_inds[1, invalid]]

        ds[varname] = (data.dims, values)
        ds[varname].attrs = data.attrs
        if "_FillValue" in ds[varname].encoding:
            del ds[varname].encoding["_FillValue"]

    # Preserve a fill value for slabs that remain fully invalid after
    # extrapolation so ncremap ignores them during interpolation.
    encoding = {}
    for varname in varnames:
        dtype = ds[varname].dtype
        if np.issubdtype(dtype, np.floating) and \
                bool(np.any(np.isnan(ds[varname].values))):
            fill = netCDF4.default_fillvals[dtype.str[1:]]
            encoding[varname] = {"_FillValue": dtype.type(fill)}

    # Write CDF-5 (NETCDF3_64BIT_DATA): a classic-model format with 64-bit
    # sizes that supports very large variables (e.g. AIS 3D ocean thermal
    # forcing) without the HDF5 chunk-size limits that make ncremap unable to
    # open large NETCDF4 files. This is a temporary file consumed by ncremap.
    for variable in ds.variables.values():
        variable.encoding = {}
    ds.to_netcdf(output_file, format="NETCDF3_64BIT_DATA", engine="netcdf4",
                 encoding=encoding)
    ds.close()


def open_rename_and_trim(remapped_file, rename_vars, start_year, end_year):
    """
    Open a remapped file, rename dimensions/variables to MALI conventions,
    and restrict to the requested year range.

    Parameters
    ----------
    remapped_file : str
        Data remapped onto the MALI mesh

    rename_vars : dict
        Mapping of source variable names to MALI variable names

    start_year : int
        First year (inclusive) to retain

    end_year : int
        Last year (inclusive) to retain

    Returns
    -------
    ds : xarray.Dataset
        The renamed and trimmed dataset

    years : numpy.ndarray
        The integer years retained
    """
    # The time coordinate has units="year" (integer years), which is not
    # CF-compliant, so disable time decoding.
    ds = xr.open_dataset(remapped_file, decode_times=False)

    # Capture integer years before the time coordinate is renamed
    years = ds["time"].values.astype(int)

    rename_dims = {}
    if "ncol" in ds.dims:
        rename_dims["ncol"] = "nCells"
    if "time" in ds.dims:
        rename_dims["time"] = "Time"
    if rename_dims:
        ds = ds.rename(rename_dims)

    rename_vars = {src: dst for src, dst in rename_vars.items() if src in ds}
    if rename_vars:
        ds = ds.rename(rename_vars)

    # Restrict to the requested year range
    keep = (years >= start_year) & (years <= end_year)
    ds = ds.isel(Time=keep)
    years = years[keep]

    return ds, years


def add_xtime_and_write(ds, years, output_file):
    """
    Add an ``xtime`` variable (January 1st of each year), drop auxiliary
    remapping variables, and write the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to finalize (annual fields applied at the start of the
        year)

    years : numpy.ndarray
        The integer years, one per Time index

    output_file : str
        Output file path
    """
    xtime = [f"{int(yr):04d}-01-01_00:00:00".ljust(64) for yr in years]
    ds["xtime"] = ("Time", xtime)
    ds["xtime"] = ds.xtime.astype("S")

    vars_to_drop = [v for v in ["lat_vertices", "lon_vertices", "lat",
                                "lon", "area", "Time"]
                    if v in ds]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    # Clear encoding to avoid inherited chunking/compression that would
    # conflict with NETCDF3_64BIT_DATA
    for variable in ds.variables.values():
        variable.encoding = {}

    # Write CDF-5 (NETCDF3_64BIT_DATA) for MALI/PIO compatibility — HDF5-based
    # netCDF-4 is not supported by the Fortran PIO reader
    ds.to_netcdf(output_file, format="NETCDF3_64BIT_DATA", engine="netcdf4")
    ds.close()
