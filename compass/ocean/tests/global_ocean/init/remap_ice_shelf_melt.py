import h5py
import numpy as np
import pyproj
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from pyremap import MpasCellMeshDescriptor, ProjectionGridDescriptor, Remapper
from scipy.spatial import KDTree

from compass.ocean.mesh.cull import write_map_culled_to_base
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
        """
        super().__init__(test_case, name='remap_ice_shelf_melt', ntasks=512,
                         min_tasks=1)

        base_mesh_path = mesh.steps['base_mesh'].path
        culled_mesh_path = mesh.steps['cull_mesh'].path

        self.add_input_file(
            filename='base_mesh.nc',
            work_dir_target=f'{base_mesh_path}/base_mesh.nc')

        self.add_input_file(
            filename='culled_mesh.nc',
            work_dir_target=f'{culled_mesh_path}/culled_mesh.nc')

        self.add_input_file(
            filename='map_culled_to_base.nc',
            work_dir_target=f'{culled_mesh_path}/map_culled_to_base.nc')

        self.add_input_file(
            filename='land_ice_mask.nc',
            work_dir_target=f'{culled_mesh_path}/land_ice_mask.nc')

        self.add_input_file(
            filename='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
            target='Paolo_2023_ANT_G1920V01_IceShelfMelt.nc',
            database='initial_condition_database',
            url='https://its-live-data.s3.amazonaws.com/height_change/Antarctica/Floating/ANT_G1920V01_IceShelfMelt.nc')    # noqa: E501

        self.add_output_file(filename='prescribed_ismf_paolo2023.nc')

        self.mesh = mesh

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        ntasks = self.ntasks

        in_filename = 'Paolo_2023_ANT_G1920V01_IceShelfMelt.nc'

        out_filename = 'prescribed_ismf_paolo2023.nc'

        parallel_executable = config.get('parallel', 'parallel_executable')

        base_mesh_filename = 'base_mesh.nc'
        culled_mesh_filename = 'culled_mesh.nc'
        land_ice_mask_filename = 'land_ice_mask.nc'
        map_culled_to_base_filename = 'map_culled_to_base.nc'
        mesh_name = self.mesh.mesh_name

        remap_paolo(
            in_filename, base_mesh_filename, culled_mesh_filename,
            mesh_name, land_ice_mask_filename, out_filename,
            logger=logger, mpi_tasks=ntasks,
            parallel_executable=parallel_executable,
            map_culled_to_base_filename=map_culled_to_base_filename)


def remap_paolo(in_filename, base_mesh_filename, culled_mesh_filename,
                mesh_name, land_ice_mask_filename, out_filename, logger,
                mapping_directory='.', method='conserve',
                renormalization_threshold=None, mpi_tasks=1,
                parallel_executable=None,
                map_culled_to_base_filename=None):
    """
    Remap the Paolo et al. (2023; https://doi.org/10.5194/tc-17-3409-2023)
    melt rates at ~2 km resolution to an MPAS mesh

    Parameters
    ----------
    in_filename : str
        The original Paolo et al. (2023) melt rates

    base_mesh_filename : str
        The base MPAS mesh before land is culled

    culled_mesh_filename : str
        The MPAS mesh after land has been culled

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

    map_culled_to_base_filename : str, optional
        A file with indices that map from the culled to the base MPAS mesh. If
        not provided, they will be computed
    """

    logger.info(f'Reading {in_filename}...')
    with xr.open_dataset(in_filename) as ds_in:

        x = ds_in.x
        y = ds_in.y
        melt_rate = ds_in.melt_mean
        melt_rate = melt_rate.where(melt_rate.notnull(), 0.)
        logger.info('done.')

    lx = np.abs(1e-3 * (x[-1] - x[0])).values
    ly = np.abs(1e-3 * (y[-1] - y[0])).values
    dx = np.abs(1e-3 * (x[1] - x[0])).values

    in_grid_name = f'{lx:.1f}x{ly:.1f}km_{dx:.3f}km_Antarctic_stereo'

    projection = pyproj.Proj('+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0 '
                             '+k_0=1.0 +x_0=0.0 +y_0=0.0 +ellps=WGS84')

    logger.info('Creating the source grid descriptor...')
    in_descriptor = ProjectionGridDescriptor.create(
        projection=projection, x=x.values, y=y.values, meshName=in_grid_name)
    logger.info('done.')

    out_descriptor = MpasCellMeshDescriptor(base_mesh_filename, mesh_name)

    mapping_filename = \
        f'{mapping_directory}/map_{in_grid_name}_to_{mesh_name}_base.nc'

    logger.info(f'Creating the mapping file {mapping_filename}...')
    remapper = Remapper(in_descriptor, out_descriptor, mapping_filename)

    remapper.build_mapping_file(method=method, mpiTasks=mpi_tasks,
                                tempdir=mapping_directory, logger=logger,
                                esmf_parallel_exec=parallel_executable,
                                include_logs=True)
    logger.info('done.')

    dx = np.abs(in_descriptor.xCorner[1:] - in_descriptor.xCorner[:-1])
    dy = np.abs(in_descriptor.yCorner[1:] - in_descriptor.yCorner[:-1])
    dx, dy = np.meshgrid(dx, dy)
    planar_area = xr.DataArray(dims=('y', 'x'), data=dx * dy)

    with xr.open_dataset(mapping_filename) as ds_map:
        earth_radius = constants['SHR_CONST_REARTH']
        map_src_area = ds_map.area_a.values * earth_radius**2
        map_dst_area = ds_map.area_b.values * earth_radius**2
        sphere_area = xr.DataArray(
            dims=('y', 'x'), data=map_src_area.reshape(planar_area.shape))

    logger.info('Creating the source xarray dataset...')
    ds = xr.Dataset()

    # convert to the units and variable names expected in MPAS-O

    # Paolo et al. (2023) ice density (from attributes)
    rho_ice = 917.
    s_per_yr = 365. * constants['SHR_CONST_CDAY']
    latent_heat_of_fusion = constants['SHR_CONST_LATICE']
    ds['x'] = x
    ds['y'] = y
    area_ratio = planar_area / sphere_area
    logger.info(f'min projected area ratio: {area_ratio.min().values}')
    logger.info(f'max projected area ratio: {area_ratio.max().values}')
    logger.info('')

    # original field is negative for melt
    fwf = -melt_rate * rho_ice / s_per_yr
    field = 'dataLandIceFreshwaterFlux'
    ds[field] = area_ratio * fwf
    ds[field].attrs['units'] = 'kg m^-2 s^-1'
    field = 'dataLandIceHeatFlux'
    ds[field] = (latent_heat_of_fusion *
                 ds.dataLandIceFreshwaterFlux)
    ds[field].attrs['units'] = 'W m^-2'
    logger.info('Writing the source dataset...')
    write_netcdf(ds, 'Paolo_2023_ismf_1992-2017_v1.0.nc')
    logger.info('done.')
    logger.info('')

    planar_flux = (fwf * planar_area).sum().values
    sphere_flux = (ds.dataLandIceFreshwaterFlux * sphere_area).sum().values

    logger.info(f'Area of a cell (m^2):             {planar_area[0,0]:.1f}')
    logger.info(f'Total flux on plane (kg/s):       {planar_flux:.1f}')
    logger.info(f'Total flux on sphere (kg/s):      {sphere_flux:.1f}')
    logger.info('')

    logger.info('Remapping...')
    ds_remap = remapper.remap(
        ds, renormalizationThreshold=renormalization_threshold)
    logger.info('done.')
    logger.info('')

    with xr.open_dataset(base_mesh_filename) as ds_mesh:
        mpas_area_cell = ds_mesh.areaCell
        sphere_area_cell = xr.DataArray(
            dims=('nCells',), data=map_dst_area)

    area_ratio = sphere_area_cell / mpas_area_cell
    logger.info(f'min MPAS area ratio: {area_ratio.min().values}')
    logger.info(f'max MPAS area ratio: {area_ratio.max().values}')
    logger.info('')

    sphere_fwf = ds_remap.dataLandIceFreshwaterFlux

    field = 'dataLandIceFreshwaterFlux'
    ds_remap[field] = area_ratio * sphere_fwf
    ds_remap[field].attrs['units'] = 'kg m^-2 s^-1'
    field = 'dataLandIceHeatFlux'
    ds_remap[field] = area_ratio * ds_remap[field]
    ds_remap[field].attrs['units'] = 'W m^-2'

    mpas_flux = (ds_remap.dataLandIceFreshwaterFlux *
                 mpas_area_cell).sum().values
    sphere_flux = (sphere_fwf * sphere_area_cell).sum().values

    logger.info(f'Total flux w/ MPAS area (kg/s):   {mpas_flux:.1f}')
    logger.info(f'Total flux w/ sphere area (kg/s): {sphere_flux:.1f}')

    if map_culled_to_base_filename is None:
        map_culled_to_base_filename = 'map_culled_to_base.nc'
        write_map_culled_to_base(base_mesh_filename=base_mesh_filename,
                                 culled_mesh_filename=culled_mesh_filename,
                                 out_filename=map_culled_to_base_filename)

    _land_ice_mask_on_base_mesh(
        base_mesh_filename=base_mesh_filename,
        land_ice_mask_filename=land_ice_mask_filename,
        map_culled_to_base_filename=map_culled_to_base_filename)

    ds_mask = xr.open_dataset('land_ice_mask_on_base.nc')
    mask = ds_mask.landIceFloatingMask
    ds_remap['landIceFloatingMask'] = mask
    ds_remap.attrs.pop('history')

    write_netcdf(ds_remap, 'ismf_remapped_to_base.nc')

    # deal with melting beyond the land-ice mask
    _reroute_missing_flux(base_mesh_filename, map_culled_to_base_filename,
                          out_filename, logger)


def remap_adusumilli(in_filename, base_mesh_filename, culled_mesh_filename,
                     mesh_name, land_ice_mask_filename, out_filename, logger,
                     mapping_directory='.', method='conserve',
                     renormalization_threshold=None, mpi_tasks=1,
                     parallel_executable=None,
                     map_culled_to_base_filename=None):
    """
    Remap the Adusumilli et al. (2020) melt rates at 0.5 km resolution to an
    MPAS mesh

    Parameters
    ----------
    in_filename : str
        The original Adusumilli et al. (2020) melt rates

    base_mesh_filename : str
        The base MPAS mesh before land is culled

    culled_mesh_filename : str
        The MPAS mesh after land has been culled

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

    map_culled_to_base_filename : str, optional
        A file with indices that map from the culled to the base MPAS mesh. If
        not provided, they will be computed
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

    out_descriptor = MpasCellMeshDescriptor(base_mesh_filename, mesh_name)

    mapping_filename = \
        f'{mapping_directory}/map_{in_grid_name}_to_{mesh_name}_base.nc'

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

    if map_culled_to_base_filename is None:
        map_culled_to_base_filename = 'map_culled_to_base.nc'
        write_map_culled_to_base(base_mesh_filename=base_mesh_filename,
                                 culled_mesh_filename=culled_mesh_filename,
                                 out_filename=map_culled_to_base_filename)

    _land_ice_mask_on_base_mesh(
        base_mesh_filename=base_mesh_filename,
        land_ice_mask_filename=land_ice_mask_filename,
        map_culled_to_base_filename=map_culled_to_base_filename)

    ds_mask = xr.open_dataset('land_ice_mask_on_base.nc')
    mask = ds_mask.landIceFloatingMask
    ds_remap['landIceFloatingMask'] = mask
    ds_remap.attrs.pop('history')

    write_netcdf(ds_remap, 'ismf_remapped_to_base.nc')

    # deal with melting beyond the land-ice mask
    _reroute_missing_flux(base_mesh_filename, map_culled_to_base_filename,
                          out_filename, logger)


def _land_ice_mask_on_base_mesh(base_mesh_filename, land_ice_mask_filename,
                                map_culled_to_base_filename):
    """ Map the land-ice mask back to the base mesh """

    ds_map = xr.open_dataset(map_culled_to_base_filename)
    map_culled_to_base = ds_map.mapCulledToBase.values

    ds_base = xr.open_dataset(base_mesh_filename)
    ncells_base = ds_base.sizes['nCells']

    ds_culled_mask = xr.open_dataset(land_ice_mask_filename)
    culled_land_ice_mask = ds_culled_mask.landIceFloatingMask.values
    base_land_ice_mask = np.zeros(ncells_base, dtype=int)
    base_land_ice_mask[map_culled_to_base] = culled_land_ice_mask
    ds_base_mask = xr.Dataset()
    ds_base_mask['landIceFloatingMask'] = ('nCells', base_land_ice_mask)

    write_netcdf(ds_base_mask, 'land_ice_mask_on_base.nc')


def _reroute_missing_flux(base_mesh_filename, map_culled_to_base_filename,
                          out_filename, logger):
    """
    For each flux cell not within an ice shelf, find the closest ice-shelf cell
    """
    ds_ismf_base = xr.open_dataset('ismf_remapped_to_base.nc')
    land_ice_mask = ds_ismf_base.landIceFloatingMask
    fwf = ds_ismf_base.dataLandIceFreshwaterFlux
    fwf = fwf.where(fwf.notnull(), 0.)
    hf = ds_ismf_base.dataLandIceHeatFlux
    hf = hf.where(hf.notnull(), 0.)
    flux_mask = fwf != 0.
    fluxes_to_reroute = np.logical_and(flux_mask,
                                       land_ice_mask != 1)

    ds_base = xr.open_dataset(base_mesh_filename)
    ncells_base = ds_base.sizes['nCells']
    base_xyz = np.zeros((ncells_base, 3))
    base_xyz[:, 0] = ds_base.xCell.values
    base_xyz[:, 1] = ds_base.yCell.values
    base_xyz[:, 2] = ds_base.zCell.values
    area = ds_base.areaCell

    land_ice_xyz = base_xyz[land_ice_mask.values == 1, :]
    land_ice_cell_indices = np.arange(ncells_base)[land_ice_mask.values == 1]

    reroute_mask = fluxes_to_reroute.astype(int)

    fw_mass_rerouted = (fwf * area).isel(nCells=fluxes_to_reroute.values)
    heat_rerouted = (hf * area).isel(nCells=fluxes_to_reroute.values)

    count = np.sum(reroute_mask.values)
    logger.info(f'Rerouting fluxes from {count} cells')
    fwf_land_ice = (land_ice_mask * fwf * area).sum().values
    logger.info(f'Captured flux (kg/s):             {fwf_land_ice:.1f}')
    fwf_rerouted = fw_mass_rerouted.sum().values
    logger.info(f'Rerouted flux (kg/s):             {fwf_rerouted:.1f}')
    fwf_total = (fwf * area).sum().values
    logger.info(f'Total flux (kg/s):                {fwf_total:.1f}')
    ds_to_route = xr.Dataset()
    ds_to_route['rerouteMask'] = reroute_mask
    write_netcdf(ds_to_route, 'route_mask.nc')

    reroute_xyz = base_xyz[fluxes_to_reroute.values, :]

    # we want to find closest land-ice cell to a given reroute cell
    tree = KDTree(land_ice_xyz)

    # todo: may need to be more sophisticated with workers based on ntasks,
    # etc.
    _, indices = tree.query(reroute_xyz, workers=-1)

    ds_map = xr.open_dataset(map_culled_to_base_filename)
    map_culled_to_base = ds_map.mapCulledToBase.values

    fwf_rerouted = land_ice_mask.values * fwf.where(fwf.notnull(), 0.).values
    hf_rerouted = land_ice_mask.values * hf.where(hf.notnull(), 0.).values
    for rerouted_index, land_ice_index in enumerate(indices):
        icell = land_ice_cell_indices[land_ice_index]
        fwf_rerouted[icell] += fw_mass_rerouted[rerouted_index] / area[icell]
        hf_rerouted[icell] += heat_rerouted[rerouted_index] / area[icell]

    # now, go from base to culled mesh
    fwf_rerouted = fwf_rerouted[map_culled_to_base]
    hf_rerouted = hf_rerouted[map_culled_to_base]

    ds_out = xr.Dataset()
    field = 'dataLandIceFreshwaterFlux'
    ds_out[field] = ('nCells', fwf_rerouted)
    ds_out[field] = ds_out[field].expand_dims(dim='Time', axis=0)
    ds_out[field].attrs['units'] = 'kg m^-2 s^-1'
    field = 'dataLandIceHeatFlux'
    ds_out[field] = ('nCells', hf_rerouted)
    ds_out[field] = ds_out[field].expand_dims(dim='Time', axis=0)
    ds_out[field].attrs['units'] = 'W m^-2'
    write_netcdf(ds_out, out_filename)

    fwf_total = (ds_out.dataLandIceFreshwaterFlux *
                 area.isel(nCells=map_culled_to_base)).sum().values
    logger.info(f'Total after rerouting (kg/s):     {fwf_total:.1f}')
    logger.info('')
