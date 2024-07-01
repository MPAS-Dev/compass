import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class NormalizeIcebergIceShelfMelt(FilesForE3SMStep):
    """
    A step for for normalizing data iceberg and ice-shelf melt rates on the
    MPAS grid to a total flux of 1.0 and staging them in ``assembled_files``
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case, name='normalize_iceberg_ice_shelf_melt',
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
        setup input files based on config options
        """
        super().setup()
        if self.with_ice_shelf_cavities:
            self.add_output_file(filename='dib_merino_2020_normalized.nc')
            self.add_output_file(filename='dismf_paolo2023_normalized.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        if not self.with_ice_shelf_cavities:
            return

        logger = self.logger

        suffix = f'{self.mesh_short_name}.{self.creation_date}'

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

        for var in ['bergFreshwaterFluxData']:
            ds_dib[var] = ds_dib[var] / total_flux

        write_netcdf(ds_dib, 'dib_merino_2020_normalized.nc')

        for var in ['dataLandIceFreshwaterFlux', 'dataLandIceHeatFlux']:
            ds_dismf[var] = ds_dismf[var] / total_flux

        write_netcdf(ds_dismf, 'dismf_paolo2023_normalized.nc')

        norm_total_dib_flux = (ds_dib.bergFreshwaterFluxData * weights *
                               area_cell).sum()

        norm_total_dismf_flux = (ds_dismf.dataLandIceFreshwaterFlux *
                                 area_cell).sum()

        norm_total_flux = norm_total_dib_flux + norm_total_dismf_flux

        logger.info(f'norm_total_dib_flux:     {norm_total_dib_flux:.3f}')
        logger.info(f'norm_total_dismf_flux:   {norm_total_dismf_flux:.3f}')
        logger.info(f'norm_total_flux:         {norm_total_flux:.3f}')
        logger.info('')

        prefix = 'Iceberg_Climatology_Merino_normalized'
        dest_filename = f'{prefix}.{suffix}.nc'

        symlink(
            os.path.abspath('dib_merino_2020_normalized.nc'),
            f'{self.ocean_inputdata_dir}/{dest_filename}')

        prefix = 'prescribed_ismf_paolo2023_normalized'
        dest_filename = f'{prefix}.{suffix}.nc'

        symlink(
            os.path.abspath('dismf_paolo2023_normalized.nc'),
            f'{self.ocean_inputdata_dir}/{dest_filename}')
