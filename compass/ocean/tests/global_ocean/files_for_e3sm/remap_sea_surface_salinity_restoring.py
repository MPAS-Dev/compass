import os

import xarray as xr
from mpas_tools.io import write_netcdf
from pyremap import LatLonGridDescriptor, MpasCellMeshDescriptor, Remapper

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class RemapSeaSurfaceSalinityRestoring(FilesForE3SMStep):
    """
    A step for for remapping sea surface salinity (SSS) from the Polar science
    center Hydrographic Climatology (PHC) to the current MPAS mesh
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
                         name='remap_sea_surface_salinity_restoring',
                         ntasks=512, min_tasks=1)

        self.add_input_file(
            filename='PHC2_salx.2004_08_03.filled_double_precision.nc',
            target='PHC2_salx.2004_08_03.filled_double_precision.nc',
            database='initial_condition_database')

        self.add_output_file(filename='sss.PHC2_monthlyClimatology.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        logger = self.logger
        config = self.config
        ntasks = self.ntasks

        in_filename = 'PHC2_salx.2004_08_03.filled_double_precision.nc'

        prefix = 'sss.PHC2_monthlyClimatology'
        suffix = f'{self.mesh_short_name}.{self.creation_date}'

        remapped_filename = f'{prefix}.nc'
        dest_filename = f'{prefix}.{suffix}.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        mesh_filename = 'restart.nc'
        mesh_short_name = self.mesh_short_name

        remap_sss(in_filename, mesh_filename, mesh_short_name,
                  remapped_filename, logger=logger, mpi_tasks=ntasks,
                  parallel_executable=parallel_executable)

        symlink(
            os.path.abspath(remapped_filename),
            f'{self.ocean_inputdata_dir}/{dest_filename}')


def remap_sss(in_filename, mesh_filename, mesh_name, out_filename, logger,
              mapping_directory='.', method='bilinear', mpi_tasks=1,
              parallel_executable=None):
    """
    Remap sea surface salinity (SSS) from the Polar science center
    Hydrographic Climatology (PHC) to the current MPAS mesh

    Parameters
    ----------
    in_filename : str
        The original PHC sea surface salinity file

    mesh_filename : str
        The MPAS mesh

    mesh_name : str
        The name of the mesh (e.g. oEC60to30wISC), used in the name of the
        mapping file

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

    mpi_tasks : int, optional
        The number of MPI tasks to use to compute the mapping file

    parallel_executable : {'srun', 'mpirun'}, optional
        The name of the parallel executable to use to launch ESMF tools.
        But default, 'mpirun' from the conda environment is used
    """

    logger.info('Creating the source grid descriptor...')
    src_descriptor = LatLonGridDescriptor.read(fileName=in_filename)
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

    logger.info('Remapping...')
    name, ext = os.path.splitext(out_filename)
    remap_filename = f'{name}_after_remap{ext}'
    remapper.remap_file(inFileName=in_filename, outFileName=remap_filename,
                        logger=logger)

    ds = xr.open_dataset(remap_filename)
    logger.info('Removing lat/lon bounds variables...')
    drop = [var for var in ds if 'nv' in ds[var].dims]
    ds = ds.drop_vars(drop)
    logger.info('Renaming dimensions and variables...')
    rename = dict(ncol='nCells',
                  SALT='surfaceSalinityMonthlyClimatologyValue')
    ds = ds.rename(rename)
    write_netcdf(ds, out_filename)

    logger.info('done.')
