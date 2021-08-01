import os
import xarray

from mpas_tools.io import write_netcdf


def remove_inactive_top_cells_output(in_filename, out_filename=None,
                                     mesh_filename=None):
    """
    Remove inactive top cells from the output netCDF file

    Parameters
    ----------
    in_filename : str
        Filename for the netCDF file to be cropped

    out_filename : str, optional
        Filename for the netCDF file after cropping.  Tbe default name is the
        original file name with ``_crop`` appended before the extension

    mesh_filename : str, optional
        Filename for an MPAS mesh if not included in the file to be cropped

    """
    if not os.path.exists(in_filename):
        raise OSError(f'File {in_filename} does not exist.')

    if out_filename is None:
        basename, ext = os.path.splitext(in_filename)
        out_filename = f'{basename}_crop{ext}'

    with xarray.open_dataset(in_filename) as ds_in:
        if mesh_filename is None:
            ds_mesh = ds_in
        else:
            ds_mesh = xarray.open_dataset(mesh_filename)
        minLevelCell = ds_mesh.minLevelCell
        minval = minLevelCell.min().values
        maxval = minLevelCell.max().values
        if minval != maxval:
            raise ValueError('Expected minLevelCell to have a constant '
                             'value for inactive top cell tests')
        ds_out = ds_in.isel(nVertLevels=slice(minval-1, None))

    write_netcdf(ds_out, out_filename)
