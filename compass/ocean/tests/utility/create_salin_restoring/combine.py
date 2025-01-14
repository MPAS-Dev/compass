import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.step import Step


class Combine(Step):
    """
    A step for combining January through December sea surface salinity
    data into a single file for salinity restoring in G-cases.
    The top level data of the monthly WOA23 is utilized.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.create_salin_restoring.CreateSalinRestoring
            The test case this step belongs to
        """  # noqa: E501
        super().__init__(test_case, name='combine', ntasks=1,
                         min_tasks=1)
        self.add_output_file(filename='woa_surface_salinity_monthly.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        base_url = \
            'https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA'

        woa_dir = 'salinity/netcdf/decav91C0/0.25'

        woa_files = dict(jan='woa23_decav91C0_s01_04.nc',
                         feb='woa23_decav91C0_s02_04.nc',
                         mar='woa23_decav91C0_s03_04.nc',
                         apr='woa23_decav91C0_s04_04.nc',
                         may='woa23_decav91C0_s05_04.nc',
                         jun='woa23_decav91C0_s06_04.nc',
                         jul='woa23_decav91C0_s07_04.nc',
                         aug='woa23_decav91C0_s08_04.nc',
                         sep='woa23_decav91C0_s09_04.nc',
                         octo='woa23_decav91C0_s10_04.nc',
                         nov='woa23_decav91C0_s11_04.nc',
                         dec='woa23_decav91C0_s12_04.nc')

        for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                      'aug', 'sep', 'octo', 'nov', 'dec']:
            woa_filename = woa_files[month]
            woa_url = f'{base_url}/{woa_dir}/{woa_filename}'

            self.add_input_file(
                filename=f'woa_salin_{month}.nc',
                target=woa_filename,
                database='initial_condition_database',
                url=woa_url)

    def run(self):
        """
        Run this step of the test case
        """
        ds_jan = xr.open_dataset('woa_salin_jan.nc', decode_times=False)

        ds_out = xr.Dataset()

        for var in ['lon', 'lat']:
            ds_out[var] = ds_jan[var]
            ds_out[f'{var}_bnds'] = ds_jan[f'{var}_bnds']

        slices = list()
        for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                      'aug', 'sep', 'octo', 'nov', 'dec']:
            ds = xr.open_dataset(
                f'woa_salin_{month}.nc',
                decode_times=False).isel(depth=0). \
                drop_vars('depth')
            slices.append(ds.s_an)

        ds_out['s_an'] = xr.concat(slices, dim='time')
        ds_out['s_an'].attrs = ds_jan['s_an'].attrs

        # Change names of variables and alter attributes
        ds_out = ds_out.rename_dims({'time': 'Time'})
        ds_out = ds_out.rename_vars({'s_an': 'SALT'})
        ds_out = ds_out.drop_vars('time')

        # Create a time index
        time_var = np.arange(1, 12.1, 1)
        ds_out = ds_out.assign(Time=xr.DataArray(time_var, dims=['Time']))

        # Create a xtime array
        xtime_list = []
        for i in range(1, 13):
            xtime_string = f"0000-{i:02d}-15_00:00:00"
            xtime_list.append(xtime_string)
        xtime_out = np.array(xtime_list, dtype='S64')

        ds_out = ds_out.assign(xtime=xr.DataArray(xtime_out, dims=['Time']))

        # Change attributes to be consistent with PHC restoring file
        ds_out.Time.attrs['long_name'] = "Month Index"
        ds_out.Time.attrs['units'] = "month"
        ds_out.Time.attrs['axis'] = "T"

        # Save file and change time dimension to unlimited
        write_netcdf(ds_out, 'woa_surface_salinity_monthly.nc')
