import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class AddTotalIcebergIceShelfMelt(FilesForE3SMStep):
    """
    A step for for adding the total data iceberg and ice-shelf melt rates to
    to the data iceberg and ice-shelf melt files and staging them in
    ``assembled_files``
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case, name='add_total_iceberg_ice_shelf_melt',
                         ntasks=1, min_tasks=1)

        filename = 'Iceberg_Climatology_Merino_MPAS.nc'
        subdir = 'remap_iceberg_climatology'
        self.add_input_file(
            filename=filename,
            target=f'../{subdir}/{filename}')

        filename = 'prescribed_ismf_paolo2023.nc'
        subdir = 'remap_ice_shelf_melt'
        self.add_input_file(
            filename=filename,
            target=f'../{subdir}/{filename}')

    def setup(self):
        """
        setup output files based on config options
        """
        super().setup()
        if self.with_ice_shelf_cavities:
            self.add_output_file(
                filename='Iceberg_Climatology_Merino_MPAS_with_totals.nc')
            self.add_output_file(
                filename='prescribed_ismf_paolo2023_with_totals.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        if not self.with_ice_shelf_cavities:
            return

        logger = self.logger

        ds_dib = xr.open_dataset('Iceberg_Climatology_Merino_MPAS.nc')
        ds_dismf = xr.open_dataset('prescribed_ismf_paolo2023.nc')
        ds_mesh = xr.open_dataset('restart.nc')

        area_cell = ds_mesh.areaCell

        days_in_month = np.array(
            [31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])

        weights = xr.DataArray(data=days_in_month / 365.,
                               dims=('Time',))

        total_dib_flux = (ds_dib.bergFreshwaterFluxData * weights *
                          area_cell).sum()

        total_dismf_flux = (ds_dismf.dataLandIceFreshwaterFlux *
                            area_cell).sum()

        total_flux = total_dib_flux + total_dismf_flux

        logger.info(f'total_dib_flux:     {total_dib_flux:.1f}')
        logger.info(f'total_dismf_flux:   {total_dismf_flux:.1f}')
        logger.info(f'total_flux:         {total_flux:.1f}')
        logger.info('')

        for ds in [ds_dib, ds_dismf]:
            ntime = ds.sizes['Time']
            field = 'areaIntegAnnMeanDataIcebergFreshwaterFlux'
            ds[field] = (('Time',), np.ones(ntime) * total_dib_flux.values)
            ds[field].attrs['units'] = 'kg s-1'
            field = 'areaIntegAnnMeanDataIceShelfFreshwaterFlux'
            ds[field] = (('Time',), np.ones(ntime) * total_dismf_flux.values)
            ds[field].attrs['units'] = 'kg s-1'
            field = 'areaIntegAnnMeanDataIcebergIceShelfFreshwaterFlux'
            ds[field] = (('Time',), np.ones(ntime) * total_flux.values)
            ds[field].attrs['units'] = 'kg s-1'

        dib_filename = 'Iceberg_Climatology_Merino_MPAS_with_totals.nc'
        write_netcdf(ds_dib, dib_filename)

        dismf_filename = 'prescribed_ismf_paolo2023_with_totals.nc'
        write_netcdf(ds_dismf, dismf_filename)

        norm_total_dib_flux = (ds_dib.bergFreshwaterFluxData * weights *
                               area_cell / total_flux).sum()

        norm_total_dismf_flux = (ds_dismf.dataLandIceFreshwaterFlux *
                                 area_cell / total_flux).sum()

        norm_total_flux = norm_total_dib_flux + norm_total_dismf_flux

        logger.info(f'norm_total_dib_flux:     {norm_total_dib_flux:.16f}')
        logger.info(f'norm_total_dismf_flux:   {norm_total_dismf_flux:.16f}')
        logger.info(f'norm_total_flux:         {norm_total_flux:.16f}')
        logger.info(f'1 - norm_total_flux:     {1 - norm_total_flux:.16g}')
        logger.info('')

        prefix = 'Iceberg_Climatology_Merino'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'
        dest_filename = f'{prefix}.{suffix}.nc'
        symlink(
            os.path.abspath(dib_filename),
            f'{self.seaice_inputdata_dir}/{dest_filename}')

        prefix = 'prescribed_ismf_paolo2023'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'
        dest_filename = f'{prefix}.{suffix}.nc'
        symlink(
            os.path.abspath(dismf_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')
