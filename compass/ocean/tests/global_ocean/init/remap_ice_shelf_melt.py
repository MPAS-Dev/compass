import h5py
import numpy as np
import pyproj
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from pyremap import MpasCellMeshDescriptor, ProjectionGridDescriptor, Remapper

from compass.step import Step


class RemapIceShelfMelt(Step):
    """
    A step for for remapping observed melt rates to the MPAS grid

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run
    """
    def __init__(self, test_case, mesh):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run
        """  # noqa: E501
        super().__init__(test_case, name='remap_ice_shelf_melt', ntasks=512,
                         min_tasks=1)

        mesh_path = mesh.get_cull_mesh_path()

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='land_ice_mask.nc',
            work_dir_target=f'{mesh_path}/land_ice_mask.nc')

        self.add_input_file(
            filename='Adusumilli_2020_iceshelf_melt_rates_2010-2018_v0.h5',
            target='Adusumilli_2020_iceshelf_melt_rates_2010-2018_v0.h5',
            database='initial_condition_database',
            url='http://library.ucsd.edu/dc/object/bb0448974g/_3_1.h5')

        self.add_output_file(filename='prescribed_ismf_adusumilli2020.nc')

        self.mesh = mesh

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        ntasks = self.ntasks

        in_filename = 'Adusumilli_2020_iceshelf_melt_rates_2010-2018_v0.h5'

        out_filename = 'prescribed_ismf_adusumilli2020.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        mesh_filename = 'mesh.nc'
        land_ice_mask_filename = 'land_ice_mask.nc'
        mesh_name = self.mesh.mesh_name

        remap_adusumilli(in_filename, mesh_filename, mesh_name,
                         land_ice_mask_filename, out_filename,
                         logger=logger, mpi_tasks=ntasks,
                         parallel_executable=parallel_executable)


def remap_adusumilli(in_filename, mesh_filename, mesh_name,
                     land_ice_mask_filename, out_filename, logger,
                     mapping_directory='.', method='conserve',
                     renormalization_threshold=None, mpi_tasks=1,
                     parallel_executable=None):
    """
    Remap the Adusumilli et al. (2020) melt rates at 1 km resolution to an MPAS
    mesh

    Parameters
    ----------
    in_filename : str
        The original Adusumilli et al. (2020) melt rates

    mesh_filename : str
        The MPAS mesh

    mesh_name : str
        The name of the mesh (e.g. oEC60to30wISC), used in the name of the
        mapping file

    land_ice_mask_filename : str
        A file containing the variable ``landIceMask`` on the MPAS mesh

    out_filename : str
        The melt rates interpolated to the MPAS mesh with ocean sensible heat
        fluxes added on (assuming insulating ice)

    logger : logging.Logger
        A logger for output from the step

    mapping_directory : str
        The directory where the mapping file should be stored (if it is to be
        computed) or where it already exists (if not)

    method : {'bilinear', 'neareststod', 'conserve'}, optional
        The method of interpolation used, see documentation for
        `ESMF_RegridWeightGen` for details.

    renormalization_threshold : float, optional
        The minimum weight of a destination cell after remapping, below
        which it is masked out, or ``None`` for no renormalization and
        masking.

    mpi_tasks : int, optional
        The number of MPI tasks to use to compute the mapping file

    parallel_executable : {'srun', 'mpirun'}, optional
        The name of the parallel executable to use to launch ESMF tools.
        But default, 'mpirun' from the conda environment is used
    """

    logger.info(f'Reading {in_filename}...')
    h5_data = h5py.File(in_filename, 'r')

    x = np.array(h5_data['/x'])[:, 0]
    y = np.array(h5_data['/y'])[:, 0]
    melt_rate = np.array(h5_data['/w_b'])
    logger.info('done.')

    lx = np.abs(1e-3 * (x[-1] - x[0]))
    ly = np.abs(1e-3 * (y[-1] - y[0]))

    in_grid_name = f'{lx}x{ly}km_0.5km_Antarctic_stereo'

    projection = pyproj.Proj('+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0 '
                             '+k_0=1.0 +x_0=0.0 +y_0=0.0 +ellps=WGS84')

    logger.info('Creating the source grid descriptor...')
    in_descriptor = ProjectionGridDescriptor.create(
        projection=projection, x=x, y=y, meshName=in_grid_name)
    logger.info('done.')

    logger.info('Creating the source xarray dataset...')
    ds = xr.Dataset()

    # convert to the units and variable names expected in MPAS-O

    # Adusumilli et al. (2020) ice density (caption of Fig. 1 and Methods
    # section)
    rho_ice = 917.
    s_per_yr = 365. * constants['SHR_CONST_CDAY']
    latent_heat_of_fusion = constants['SHR_CONST_LATICE']
    mask = np.isfinite(melt_rate)
    melt_rate = np.where(mask, melt_rate, 0.)
    ds['x'] = (('x',), x)
    ds['y'] = (('y',), y)
    ds['dataLandIceFreshwaterFlux'] = (('y', 'x'),
                                       melt_rate * rho_ice / s_per_yr)
    ds['dataLandIceHeatFlux'] = (latent_heat_of_fusion *
                                 ds.dataLandIceFreshwaterFlux)
    logger.info('Writing the source dataset...')
    write_netcdf(ds, 'Adusumilli_2020_ismf_2010-2018_v0.nc')
    logger.info('done.')

    out_descriptor = MpasCellMeshDescriptor(mesh_filename, mesh_name)

    mapping_filename = \
        f'{mapping_directory}/map_{in_grid_name}_to_{mesh_name}.nc'

    logger.info(f'Creating the mapping file {mapping_filename}...')
    remapper = Remapper(in_descriptor, out_descriptor, mapping_filename)

    remapper.build_mapping_file(method=method, mpiTasks=mpi_tasks,
                                tempdir=mapping_directory, logger=logger,
                                esmf_parallel_exec=parallel_executable)
    logger.info('done.')

    logger.info('Remapping...')
    ds_remap = remapper.remap(
        ds, renormalizationThreshold=renormalization_threshold)
    logger.info('done.')

    ds_mask = xr.open_dataset(land_ice_mask_filename)
    mask = ds_mask.landIceMask

    for field in ['dataLandIceFreshwaterFlux',
                  'dataLandIceHeatFlux']:
        # zero out the field where it's currently NaN, and mask only to
        # regions with land ice
        ds_remap[field] = \
            mask * ds_remap[field].where(ds_remap[field].notnull(), 0.)

        # add a time dimension
        if 'Time' not in ds_remap[field].dims:
            ds_remap[field] = ds_remap[field].expand_dims(dim='Time', axis=0)

    # deal with melting beyond the land-ice mask

    ds_remap.attrs.pop('history')

    write_netcdf(ds_remap, out_filename)
