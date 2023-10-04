import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from pyremap import LatLonGridDescriptor, MpasCellMeshDescriptor, Remapper

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class RemapIcebergClimatology(FilesForE3SMStep):
    """
    A step for for remapping iceberg freshwater fluxes from a climatology from
    Merino et al. (2016) to the current MPAS mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case,
                         name='remap_iceberg_climatology',
                         ntasks=512, min_tasks=1)

        self.add_input_file(
            filename='Iceberg_Interannual_Merino.nc',
            target='Iceberg_Interannual_Merino.nc',
            database='initial_condition_database')

        self.add_output_file(filename='Iceberg_Climatology_Merino_MPAS.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        logger = self.logger
        config = self.config
        ntasks = self.ntasks

        in_filename = 'Iceberg_Interannual_Merino.nc'

        prefix = 'Iceberg_Climatology_Merino'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'

        remapped_filename = f'{prefix}_MPAS.nc'
        dest_filename = f'{prefix}.{suffix}.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        mesh_filename = 'restart.nc'
        mesh_short_name = self.mesh_short_name
        if self.with_ice_shelf_cavities:
            land_ice_mask_filename = 'initial_state.nc'
        else:
            land_ice_mask_filename = None

        remap_iceberg_climo(in_filename, mesh_filename, mesh_short_name,
                            land_ice_mask_filename, remapped_filename,
                            logger=logger, mpi_tasks=ntasks,
                            parallel_executable=parallel_executable)

        symlink(
            os.path.abspath(remapped_filename),
            f'{self.seaice_inputdata_dir}/{dest_filename}')


def remap_iceberg_climo(in_filename, mesh_filename, mesh_name,
                        land_ice_mask_filename, out_filename, logger,
                        mapping_directory='.', method='conserve', mpi_tasks=1,
                        parallel_executable=None):
    """
    Remap iceberg freshwater fluxes from a climatology from
    Merino et al. (2016) to the current MPAS mesh

    Parameters
    ----------
    in_filename : str
        The original Merino et al. iceberg freshwater flux climatology file

    mesh_filename : str
        The MPAS mesh

    mesh_name : str
        The name of the mesh (e.g. oEC60to30wISC), used in the name of the
        mapping file

    land_ice_mask_filename : str
        A file containing the variable ``landIceMask`` on the MPAS mesh

    out_filename : str
        The iceberg freshwater fluxes remapped to the MPAS mesh

    logger : logging.Logger
        A logger for output from the step

    mapping_directory : str
        The directory where the mapping file should be stored (if it is to be
        computed) or where it already exists (if not)

    method : {'bilinear', 'neareststod', 'conserve'}, optional
        The method of interpolation used, see documentation for
        `ESMF_RegridWeightGen` for details.

    mpi_tasks : int, optional
        The number of MPI tasks to use to compute the mapping file

    parallel_executable : {'srun', 'mpirun'}, optional
        The name of the parallel executable to use to launch ESMF tools.
        But default, 'mpirun' from the conda environment is used
    """

    name, ext = os.path.splitext(in_filename)
    monotonic_filename = f'{name}_monotonic_lon{ext}'

    ds = xr.open_dataset(in_filename)
    # latitude and longitude are actually 1D
    ds['lon'] = ds.longitude.isel(y=0)
    ds['lat'] = ds.latitude.isel(x=0)
    ds = ds.drop_vars(['latitude', 'longitude'])
    ds = ds.rename(dict(x='lon', y='lat'))
    # the first and last longitudes are zeroed out!!!
    ds = ds.isel(lon=slice(1, ds.sizes['lon'] - 1))

    lon_indices = np.argsort(ds.lon)
    # make sure longitudes are unique
    lon = ds.lon.isel(lon=lon_indices).values
    lon, unique_indices = np.unique(lon, return_index=True)
    lon_indices = lon_indices[unique_indices]
    ds = ds.isel(lon=lon_indices)

    ds.to_netcdf(monotonic_filename)
    logger.info('Creating the source grid descriptor...')
    src_descriptor = LatLonGridDescriptor.read(fileName=monotonic_filename)
    src_mesh_name = src_descriptor.meshName

    logger.info('Creating the destination MPAS mesh descriptor...')
    dst_descriptor = MpasCellMeshDescriptor(mesh_filename, mesh_name)

    mapping_filename = \
        f'{mapping_directory}/map_{src_mesh_name}_to_{mesh_name}_{method}.nc'

    logger.info(f'Creating the mapping file {mapping_filename}...')
    remapper = Remapper(src_descriptor, dst_descriptor, mapping_filename)

    remapper.build_mapping_file(method=method, mpiTasks=mpi_tasks,
                                tempdir=mapping_directory, logger=logger,
                                esmf_parallel_exec=parallel_executable)

    logger.info('Remapping...')
    name, ext = os.path.splitext(out_filename)
    remap_filename = f'{name}_after_remap{ext}'
    remapper.remap_file(inFileName=monotonic_filename,
                        outFileName=remap_filename,
                        logger=logger)

    ds = xr.open_dataset(remap_filename)
    logger.info('Removing lat/lon vertex variables...')
    drop = [var for var in ds if 'nv' in ds[var].dims]
    ds = ds.drop_vars(drop)
    logger.info('Renaming dimensions and variables...')
    rename = dict(ncol='nCells',
                  month='Time',
                  Icb_flux='bergFreshWaterFluxData')
    ds = ds.rename(rename)
    logger.info('Adding xtime...')
    xtime = []
    for time_index in range(ds.sizes['Time']):
        time_str = f'0000-{time_index + 1:02d}-15_00:00:00'
        xtime.append(time_str)
    ds['xtime'] = ('Time', np.array(xtime, 'S64'))

    logger.info('Fix masking...')
    field = 'bergFreshWaterFluxData'
    # zero out the field where it's currently NaN
    ds[field] = ds[field].where(ds[field].notnull(), 0.)

    if land_ice_mask_filename is not None:
        logger.info('Masking out regions under ice shelves...')
        ds_mask = xr.open_dataset(land_ice_mask_filename)
        no_land_ice_mask = 1 - ds_mask.landIceMask
        if 'Time' in no_land_ice_mask.dims:
            no_land_ice_mask = no_land_ice_mask.isel(Time=0, drop=True)

        #  mask only to regions without land ice
        ds[field] = ds[field] * no_land_ice_mask

    logger.info(f'Writing to {out_filename}...')
    write_netcdf(ds, out_filename)

    logger.info('done.')
