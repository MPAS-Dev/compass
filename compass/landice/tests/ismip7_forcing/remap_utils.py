"""
Shared helpers for remapping ISMIP7 forcing data to the MALI mesh.
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
