import os

import pyproj
import xarray as xr
from mpas_tools.io import write_netcdf
from pyremap import MpasCellMeshDescriptor, ProjectionGridDescriptor, Remapper

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class RemapTidalMixing(FilesForE3SMStep):
    """
    A step for for remapping the RMS tidal velocity from the CATS model
    (https://www.usap-dc.org/view/dataset/601235) to the current MPAS mesh
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
                         name='remap_tidal_mixing',
                         ntasks=512, min_tasks=1)

        self.add_input_file(
            filename='ustar_CATS2008_S71W70.nc',
            target='ustar_CATS2008_S71W70.nc',
            database='tidal_mixing')

        self.add_output_file(filename='velocityTidalRMS_CATS2008.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        logger = self.logger
        config = self.config
        ntasks = self.ntasks

        in_filename = 'ustar_CATS2008_S71W70.nc'

        prefix = 'velocityTidalRMS_CATS2008'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'

        remapped_filename = f'{prefix}.nc'
        dest_filename = f'{prefix}.{suffix}.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        mesh_filename = 'restart.nc'
        mesh_short_name = self.mesh_short_name
        land_ice_mask_filename = 'initial_state.nc'

        remap_tidal(in_filename, mesh_filename, mesh_short_name,
                    land_ice_mask_filename, remapped_filename, logger=logger,
                    mpi_tasks=ntasks, parallel_executable=parallel_executable)

        symlink(
            os.path.abspath(remapped_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')


def remap_tidal(in_filename, mesh_filename, mesh_name, land_ice_mask_filename,
                out_filename, logger, mapping_directory='.', method='bilinear',
                renormalize=0.01, mpi_tasks=1, parallel_executable=None):
    """
    Remap the RMS tidal velocity from the CATS model
    (https://www.usap-dc.org/view/dataset/601235) to the current MPAS mesh

    Parameters
    ----------
    in_filename : str
        The original CATS tidal friction velocity file

    mesh_filename : str
        The MPAS mesh

    mesh_name : str
        The name of the mesh (e.g. oEC60to30wISC), used in the name of the
        mapping file

    land_ice_mask_filename : str
        A file containing the variable ``landIceMask`` on the MPAS mesh

    out_filename : str
        An output file to write the remapped climatology of SSS to

    logger : logging.Logger
        A logger for output from the step

    mapping_directory : str
        The directory where the mapping file should be stored (if it is to be
        computed) or where it already exists (if not)

    method : {'bilinear', 'neareststod', 'conserve'}, optional
        The method of interpolation used, see documentation for
        `ESMF_RegridWeightGen` for details.

    renormalize : float, optional
        A threshold to use to renormalize the data

    mpi_tasks : int, optional
        The number of MPI tasks to use to compute the mapping file

    parallel_executable : {'srun', 'mpirun'}, optional
        The name of the parallel executable to use to launch ESMF tools.
        But default, 'mpirun' from the conda environment is used
    """

    logger.info('Creating the source grid descriptor...')
    projection_in = pyproj.Proj('+proj=stere +lat_ts=-71.0 +lat_0=-90 '
                                '+lon_0=-70.0 +k_0=1.0 +x_0=0.0 +y_0=0.0 '
                                '+ellps=WGS84')
    in_grid_name = 'S71W70_CATS_ustar'

    src_descriptor = ProjectionGridDescriptor.read(
        projection=projection_in, fileName=in_filename, meshName=in_grid_name,
        xVarName='x', yVarName='y')
    src_mesh_name = src_descriptor.meshName

    dst_descriptor = MpasCellMeshDescriptor(mesh_filename, mesh_name)

    mapping_filename = \
        f'{mapping_directory}/map_{src_mesh_name}_to_{mesh_name}_{method}.nc'

    logger.info(f'Creating the mapping file {mapping_filename}...')
    remapper = Remapper(src_descriptor, dst_descriptor, mapping_filename)

    remapper.build_mapping_file(method=method, mpiTasks=mpi_tasks,
                                tempdir=mapping_directory, logger=logger,
                                esmf_parallel_exec=parallel_executable)
    logger.info('done.')

    field = 'velocityTidalRMS'

    logger.info('Remapping...')
    name, ext = os.path.splitext(out_filename)
    remap_filename = f'{name}_after_remap{ext}'
    remapper.remap_file(inFileName=in_filename, outFileName=remap_filename,
                        logger=logger)

    ds_mask = xr.open_dataset(land_ice_mask_filename)
    land_ice_mask = ds_mask.landIceMask

    logger.info('Renormalize and masking to ice-shelf cavities...')
    ds = xr.Dataset()
    with xr.open_dataset(mesh_filename) as ds_mesh:
        ds['lonCell'] = ds_mesh.lonCell
        ds['latCell'] = ds_mesh.latCell

    with xr.open_dataset(remap_filename) as ds_remap:
        logger.info('Renaming dimensions and variables...')
        rename = dict(ncol='nCells')
        ds_remap = ds_remap.rename(rename)
        ds_remap = ds_remap.drop_vars(['x', 'y'])

        # renormalize
        mask = ds_remap.mask > renormalize
        ds[field] = (ds_remap[field] / ds_remap.mask).where(mask, 0.)

        # mask only to regions with land ice
        ds[field] = land_ice_mask * ds[field]

        # drop Time dimension
        if 'Time' in ds[field].dims:
            ds[field] = ds[field].isel(Time=0)

        ds[field].attrs['units'] = 'm^2 s^-2'

    write_netcdf(ds, out_filename)

    logger.info('done.')
