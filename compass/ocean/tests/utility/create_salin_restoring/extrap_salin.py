from datetime import datetime

import numpy as np
import xarray as xr
from scipy.signal import convolve2d

from compass.step import Step


class ExtrapSalin(Step):
    """
    Extrapolate WOA 2023 monthly sea surface salinity data into missing ocean
    regions, including ice cavities and coasts

    Attributes
    ----------
    woa_filename : str
        The name of the output file name after extrapolation

    """
    def __init__(self, test_case):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.create_salin_restoring.
        CreateSalinRestoring
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='extrap', cpus_per_task=64,
                         min_cpus_per_task=1, openmp_threads=1)

        self.add_input_file(
            filename='woa_surface_salinity_monthly.nc',
            target='../salinity_restoring/woa_surface_salinity_monthly.nc')

        self.woa_filename = None

    def setup(self):
        """
        Determine the output filename
        """

        now = datetime.now()

        datestring = now.strftime("%Y%m%d")

        self.woa_filename = f'woa23_decav_0.25_sss_monthly_extrap.\
                              {datestring}.nc'
        self.add_output_file(self.woa_filename)

    def run(self):
        """
        Extrapolate WOA 2023 model temperature and salinity into ice-shelf
        cavities.
        """
        # extrapolate horizontally using the ocean mask
        _extrap(self.woa_filename)


def _extrap(out_filename):

    in_filename = 'woa_surface_salinity_monthly.nc'
    ds = xr.open_dataset(in_filename)

    field = ds.SALT.values.copy()

    # a small averaging kernel
    x = np.arange(-1, 2)
    x, y = np.meshgrid(x, x)
    kernel = np.exp(-0.5 * (x**2 + y**2))

    threshold = 0.01
    nlon = field.shape[-1]

    lon_with_halo = np.array([nlon - 2, nlon - 1] + list(range(nlon)) + [0, 1])
    lon_no_halo = list(range(2, nlon + 2))

    for i in range(12):
        valid = np.isfinite(field[i, :, :])
        orig_mask = valid
        prev_fill_count = 0
        while True:
            valid_weight_sum = _extrap_with_halo(valid, kernel, valid,
                                                 lon_with_halo, lon_no_halo)

            new_valid = valid_weight_sum > threshold

            # don't want to overwrite original data but do want ot smooth
            # extrapolated data
            fill_mask = np.logical_and(new_valid, np.logical_not(orig_mask))

            fill_count = np.count_nonzero(fill_mask)
            if fill_count == prev_fill_count:
                # no change so we're done
                break

            field_extrap = _extrap_with_halo(field[i, :, :], kernel, valid,
                                             lon_with_halo, lon_no_halo)
            field[i, fill_mask] = field_extrap[fill_mask] / \
                valid_weight_sum[fill_mask]

            valid = new_valid
            prev_fill_count = fill_count

    attrs = ds.SALT.attrs
    dims = ds.SALT.dims
    ds['SALT'] = (dims, field)
    ds['SALT'].attrs = attrs

    ds.to_netcdf(out_filename)


def _extrap_with_halo(field, kernel, valid, lon_with_halo, lon_no_halo):
    field = field.copy()
    field[np.logical_not(valid)] = 0.
    field_with_halo = field[:, lon_with_halo]
    field_extrap = convolve2d(field_with_halo, kernel, mode='same')
    field_extrap = field_extrap[:, lon_no_halo]
    return field_extrap
