import os
import xarray

from mpas_tools.io import write_netcdf


def remove_inactive_top_cells_output(in_filename, inactive_top_cells=1):
    """
    Remove inactive top cells from the output netCDF file

    Parameters
    ----------
    in_filename : str
        Filename for the netCDF file to be cropped

    inactive_top_cells : str, optional
        The number of inactive top cell layers. It should be equal to
        ``config.inactive_top_cells``.
    """
    if not os.path.exists(in_filename):
        raise OSError('File {} does not exist.'.format(in_filename))

    out_filename = in_filename.split('.')[0] + '_crop.nc'

    with xarray.open_dataset(in_filename) as ds_in:
        ds_out = ds_in.isel(nVertLevels=slice(1, None))

    write_netcdf(ds_out, out_filename)
