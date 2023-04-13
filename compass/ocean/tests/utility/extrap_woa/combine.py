import xarray as xr

from compass.step import Step


class Combine(Step):
    """
    A step for combining a January and an annual WOA 2023 climatology into
    a single file, where the January data is used where available and the
    annual in the deeper ocean where monthly data is not provided.
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.extrap_woa.ExtraWoa
            The test case this step belongs to
        """
        super().__init__(test_case, name='combine', ntasks=1, min_tasks=1)
        self.add_output_file(filename='woa_combined.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        base_url = \
            'https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA'

        dirs = dict(
            temp=dict(ann='temperature/netcdf/decav91C0/0.25',
                      jan='temperature/netcdf/decav91C0/0.25'),
            salin=dict(ann='salinity/netcdf/decav91C0/0.25',
                       jan='salinity/netcdf/decav91C0/0.25'))

        woa_files = dict(
            temp=dict(ann='woa23_decav91C0_t00_04.nc',
                      jan='woa23_decav91C0_t01_04.nc'),
            salin=dict(
                ann='woa23_decav91C0_s00_04.nc',
                jan='woa23_decav91C0_s01_04.nc'))

        for field in ['temp', 'salin']:
            for season in ['jan', 'ann']:
                woa_dir = dirs[field][season]
                woa_filename = woa_files[field][season]
                woa_url = f'{base_url}/{woa_dir}/{woa_filename}'

                self.add_input_file(
                    filename=f'woa_{field}_{season}.nc',
                    target=woa_filename,
                    database='initial_condition_database',
                    url=woa_url)

    def run(self):
        """
        Run this step of the test case
        """
        ds_ann = xr.open_dataset('woa_temp_ann.nc', decode_times=False)

        ds_out = xr.Dataset()

        for var in ['lon', 'lat', 'depth']:
            ds_out[var] = ds_ann[var]
            ds_out[f'{var}_bnds'] = ds_ann[f'{var}_bnds']

        var_map = dict(temp='t_an', salin='s_an')

        for field, var in var_map.items():
            slices = list()
            ds_ann = xr.open_dataset(
                f'woa_{field}_ann.nc',
                decode_times=False).isel(time=0).drop_vars('time')
            ds_jan = xr.open_dataset(
                f'woa_{field}_jan.nc',
                decode_times=False).isel(time=0).drop_vars('time')
            for index in range(ds_ann.sizes['depth']):
                if index < ds_jan.sizes['depth']:
                    ds = ds_jan
                else:
                    ds = ds_ann
                slices.append(ds[var].isel(depth=index))

            ds_out[var] = xr.concat(slices, dim='depth')
            ds_out[var].attrs = ds_ann[var].attrs

        ds_out.to_netcdf('woa_combined.nc')
