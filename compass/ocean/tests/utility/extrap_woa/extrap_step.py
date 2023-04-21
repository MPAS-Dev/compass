import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import progressbar
import xarray as xr
from scipy.signal import convolve2d

from compass.step import Step


class ExtrapStep(Step):
    """
    Extrapolate WOA 2023 data into missing ocean regions, then land and
    grounded ice

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
        test_case : compass.ocean.tests.utility.extrap_woa.ExtrapWoa
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='extrap', cpus_per_task=64,
                         min_cpus_per_task=1, openmp_threads=1)

        self.add_input_file(
            filename='woa.nc',
            target='../combine/woa_combined.nc')

        self.add_input_file(
            filename='topography.nc',
            target='../remap_topography/topography_remapped.nc')

        self.woa_filename = None

    def setup(self):
        """
        Determine the output filename
        """
        self.woa_filename = 'woa23_decav_0.25_extrap.nc'
        self.add_output_file(self.woa_filename)

    def run(self):
        """
        Extrapolate WOA 2023 model temperature and salinity into ice-shelf
        cavities.
        """
        pool = Pool(self.cpus_per_task)

        self._make_3d_ocean_mask()

        # extrapolate horizontally using the ocean mask
        self._extrap_horiz(use_ocean_mask=True, pool=pool)

        # extrapolate vertically using the ocean mask
        self._extrap_vert(use_ocean_mask=True)

        # extrapolate horizontally into land and grounded ice
        self._extrap_horiz(use_ocean_mask=False, pool=pool)

        # extrapolate vertically into land and grounded ice
        self._extrap_vert(use_ocean_mask=False)

        pool.terminate()

    @staticmethod
    def _make_3d_ocean_mask():
        grid_filename = 'woa.nc'
        topo_filename = 'topography.nc'
        out_filename = 'ocean_mask.nc'

        with xr.open_dataset(topo_filename) as ds_topo:
            bathymetry = ds_topo.bathymetry
            ocean_mask = ds_topo.ocean_mask

            ds_out = xr.Dataset()
            with xr.open_dataset(grid_filename) as ds_grid:
                for var in ['lon', 'lat', 'depth']:
                    ds_out[var] = ds_grid[var]
                    ds_out[f'{var}_bnds'] = ds_grid[f'{var}_bnds']

                z_top = -ds_grid.depth_bnds.isel(nbounds=0)

                ocean_mask_3d = np.logical_and(
                    bathymetry <= z_top,
                    ocean_mask >= 0.5).astype(int)

                ocean_mask_3d = \
                    ocean_mask_3d.transpose('depth', 'lat', 'lon')

                ds_out['ocean_mask'] = ocean_mask_3d

                ds_out.to_netcdf(out_filename)

    def _extrap_horiz(self, use_ocean_mask, pool):
        logger = self.logger

        if use_ocean_mask:
            in_filename = 'woa.nc'
            out_filename = 'extrap_ocean/woa_extrap_horiz.nc'
            progress_dir = 'extrap_ocean/extrap_horiz'
        else:
            in_filename = 'extrap_ocean/woa_extrap.nc'
            out_filename = 'extrap_land/woa_extrap_horiz.nc'
            progress_dir = 'extrap_land/extrap_horiz'

        try:
            os.makedirs(progress_dir)
        except FileExistsError:
            pass

        with xr.open_dataset(in_filename) as ds_woa:
            ndepth = ds_woa.sizes['depth']
            dims = ds_woa.pt_an.dims

        logger.info('  Horizontally extrapolating WOA data...')
        progress = self.log_filename is None

        if progress:
            widgets = ['  ', progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=ndepth).start()
        else:
            bar = None

        partial_func = partial(_extrap_level, use_ocean_mask, in_filename,
                               progress_dir)

        depth_indices = range(ndepth)
        files = list()
        for depth_index, tmp_filename in enumerate(
                pool.imap(partial_func, depth_indices)):
            files.append(tmp_filename)
            if progress:
                bar.update(depth_index + 1)

        if progress:
            bar.finish()

        ds_out = xr.open_mfdataset(files, combine='nested', concat_dim='depth')
        for field_name in ['pt_an', 's_an']:
            ds_out[field_name] = ds_out[field_name].transpose(*dims)
        ds_out.to_netcdf(out_filename)

    def _extrap_vert(self, use_ocean_mask):
        logger = self.logger

        if use_ocean_mask:
            in_filename = 'extrap_ocean/woa_extrap_horiz.nc'
            out_filename = 'extrap_ocean/woa_extrap.nc'
        else:
            in_filename = 'extrap_land/woa_extrap_horiz.nc'
            out_filename = 'woa23_decav_0.25_extrap.nc'

        ds = xr.open_dataset(in_filename)

        if use_ocean_mask:
            ds_mask = xr.open_dataset('ocean_mask.nc')
            ocean_mask = ds_mask.ocean_mask
        else:
            ocean_mask = None

        ndepth = ds.sizes['depth']

        logger.info('  Vertically extrapolating WOA data...')
        progress = self.log_filename is None
        if progress:
            widgets = [f'  pt_an z=1/{ndepth}: ',
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=2 * ndepth).start()
        else:
            bar = None

        count = 0
        for field_name in ['pt_an', 's_an']:
            slices = [ds[field_name].isel(depth=0).drop_vars(['depth'])]
            for depth_index in range(1, ndepth):
                field = ds[field_name]
                field_above = field.isel(depth=depth_index - 1)
                field_local = field.isel(depth=depth_index)
                mask = field_local.isnull()
                if ocean_mask is not None:
                    mask = np.logical_and(mask,
                                          ocean_mask.isel(depth=depth_index))
                field_local = xr.where(mask, field_above, field_local)
                slices.append(field_local)

                count += 1
                if progress:
                    bar.widgets[0] = \
                        f'  {field_name} z={depth_index + 1}/{ndepth}: '
                    bar.update(count)

            field = xr.concat(slices, dim='depth')
            attrs = ds[field_name].attrs
            dims = ds[field_name].dims
            ds[field_name] = field.transpose(*dims)
            ds[field_name].attrs = attrs

        if progress:
            bar.finish()

        ds.to_netcdf(out_filename)


def _extrap_level(use_ocean_mask, in_filename, progress_dir, depth_index):

    out_filename = os.path.join(progress_dir, f'woa_lev_{depth_index}.nc')
    ds = xr.open_dataset(in_filename).isel(depth=depth_index)

    if use_ocean_mask:
        ds_mask = xr.open_dataset('ocean_mask.nc').isel(depth=depth_index)
        ocean_mask = ds_mask.ocean_mask.values
    else:
        ocean_mask = None

    field = ds.pt_an.values

    # a small averaging kernel
    x = np.arange(-1, 2)
    x, y = np.meshgrid(x, x)
    kernel = np.exp(-0.5 * (x**2 + y**2))

    # a threshold for extrapolation weights to be considered valid
    threshold = 0.01

    valid = np.isfinite(field)
    orig_mask = valid
    if ocean_mask is not None:
        invalid_after_fill = np.logical_not(np.logical_or(valid, ocean_mask))
    else:
        invalid_after_fill = None

    fields = dict(pt_an=ds.pt_an.values.copy(),
                  s_an=ds.s_an.values.copy())

    nlon = field.shape[1]

    lon_with_halo = np.array([nlon - 2, nlon - 1] + list(range(nlon)) + [0, 1])
    lon_no_halo = list(range(2, nlon + 2))

    prev_fill_count = 0

    while True:
        valid_weight_sum = _extrap_with_halo(valid, kernel, valid,
                                             lon_with_halo, lon_no_halo)
        if invalid_after_fill is not None:
            valid_weight_sum[invalid_after_fill] = 0.

        new_valid = valid_weight_sum > threshold

        # don't want to overwrite original data but do want ot smooth
        # extrapolated data
        fill_mask = np.logical_and(new_valid, np.logical_not(orig_mask))

        fill_count = np.count_nonzero(fill_mask)
        if fill_count == prev_fill_count:
            # no change so we're done
            break

        for field_name, field in fields.items():
            field_extrap = _extrap_with_halo(field, kernel, valid,
                                             lon_with_halo, lon_no_halo)

            field[fill_mask] = \
                field_extrap[fill_mask] / valid_weight_sum[fill_mask]

        valid = new_valid
        prev_fill_count = fill_count

    for field_name, field in fields.items():
        if invalid_after_fill is not None:
            field[invalid_after_fill] = np.nan
        attrs = ds[field_name].attrs
        dims = ds[field_name].dims
        ds[field_name] = (dims, field)
        ds[field_name].attrs = attrs

    ds.to_netcdf(out_filename)
    return out_filename


def _extrap_with_halo(field, kernel, valid, lon_with_halo, lon_no_halo):
    field = field.copy()
    field[np.logical_not(valid)] = 0.
    field_with_halo = field[:, lon_with_halo]
    field_extrap = convolve2d(field_with_halo, kernel, mode='same')
    field_extrap = field_extrap[:, lon_no_halo]
    return field_extrap
