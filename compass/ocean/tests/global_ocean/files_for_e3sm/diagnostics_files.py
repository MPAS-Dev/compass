import os
import xarray
import glob
import pyproj
import numpy

from pyremap import get_lat_lon_descriptor, ProjectionGridDescriptor, \
    MpasMeshDescriptor, Remapper
from geometric_features import GeometricFeatures
from geometric_features.aggregation import get_aggregator_by_name
from mpas_tools.logging import check_call
from mpas_tools.ocean.moc import add_moc_southern_boundary_transects
from mpas_tools.io import write_netcdf
import mpas_tools.io

from compass.io import symlink
from compass.step import Step


class DiagnosticsFiles(Step):
    """
    A step for creating files needed for the Meridional Overturning Circulation
    analysis member and diagnostics from MPAS-Analysis

    Attributes
    ----------
    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities
    """

    def __init__(self, test_case, restart_filename, with_ice_shelf_cavities):
        """
        Create a step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, name='diagnostics_files', cpus_per_task=18,
                         min_cpus_per_task=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target=f'../{restart_filename}')

        self.with_ice_shelf_cavities = with_ice_shelf_cavities

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']

        make_diagnostics_files(self.config, self.logger, mesh_short_name,
                               self.with_ice_shelf_cavities,
                               self.cpus_per_task)


def make_diagnostics_files(config, logger, mesh_short_name,
                           with_ice_shelf_cavities, cores):
    """
    Run this step of the testcase

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step

    mesh_short_name : str
        The E3SM short name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh has ice-shelf cavities

    cores : int
        The number of cores to use to build mapping files
    """

    for directory in [
            f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}',
            '../assembled_files/diagnostics/mpas_analysis/region_masks',
            '../assembled_files/diagnostics/mpas_analysis/maps']:
        try:
            os.makedirs(directory)
        except OSError:
            pass
    _make_moc_masks(mesh_short_name, logger, cores)

    gf = GeometricFeatures()
    region_groups = ['Antarctic Regions', 'Arctic Ocean Regions',
                     'Arctic Sea Ice Regions', 'Ocean Basins',
                     'Ocean Subbasins', 'ISMIP6 Regions']

    if with_ice_shelf_cavities:
        region_groups.append('Ice Shelves')

    for region_group in region_groups:
        function, prefix, date = get_aggregator_by_name(region_group)
        suffix = f'{prefix}{date}'
        fc_mask = function(gf)
        _make_region_masks(mesh_short_name, suffix=suffix, fc_mask=fc_mask,
                           logger=logger, cores=cores)

    transect_groups = ['Transport Transects']
    for transect_group in transect_groups:
        function, prefix, date = get_aggregator_by_name(transect_group)
        suffix = f'{prefix}{date}'
        fc_mask = function(gf)
        _make_transect_masks(mesh_short_name, suffix=suffix, fc_mask=fc_mask,
                             logger=logger, cores=cores)

    _make_analysis_lat_lon_map(config, mesh_short_name, cores, logger)
    for projection_name in ['antarctic', 'arctic', 'antarctic_extended',
                            'arctic_extended', 'north_atlantic',
                            'north_pacific', 'subpolar_north_atlantic']:
        _make_analysis_projection_map(config, mesh_short_name, projection_name,
                                      cores, logger)

    # make links in output directory
    files = glob.glob('map_*')

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/maps'
    for filename in files:
        symlink(f'../../../../diagnostics_files/{filename}',
                f'{output_dir}/{filename}')


def _make_region_masks(mesh_name, suffix, fc_mask, logger, cores):
    mesh_filename = 'restart.nc'

    geojson_filename = f'{suffix}.geojson'
    mask_filename = f'{mesh_name}_{suffix}.nc'

    fc_mask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_region_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'cell',
            '--process_count', f'{cores}',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink(f'../../../../diagnostics_files/{mask_filename}',
            f'{output_dir}/{mask_filename}')


def _make_transect_masks(mesh_name, suffix, fc_mask, logger, cores,
                         subdivision_threshold=10e3):
    mesh_filename = 'restart.nc'

    geojson_filename = f'{suffix}.geojson'
    mask_filename = f'{mesh_name}_{suffix}.nc'

    fc_mask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_transect_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'edge',
            '-s', f'{subdivision_threshold}',
            '--process_count', f'{cores}',
            '--add_edge_sign',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink(f'../../../../diagnostics_files/{mask_filename}',
            f'{output_dir}/{mask_filename}')


def _make_analysis_lat_lon_map(config, mesh_name, cores, logger):
    mesh_filename = 'restart.nc'

    in_descriptor = MpasMeshDescriptor(mesh_filename, mesh_name)

    lat_res = config.getfloat('files_for_e3sm', 'comparisonLatResolution')
    lon_res = config.getfloat('files_for_e3sm', 'comparisonLonResolution')

    # modify the resolution of the global lat-lon grid as desired
    out_descriptor = get_lat_lon_descriptor(dLon=lat_res,
                                            dLat=lon_res)
    out_grid_name = out_descriptor.meshName

    _make_mapping_file(mesh_name, out_grid_name, in_descriptor, out_descriptor,
                       cores, config, logger)


# copied from MPAS-Analysis for now
def _get_pyproj_projection(comparison_grid_name):
    """
    Get the projection from the comparison_grid_name.
    Parameters
    ----------
    comparison_grid_name : str
        The name of the projection comparison grid to use for remapping
    Returns
    -------
    projection : pyproj.Proj
        The projection
    Raises
    ------
    ValueError
        If comparison_grid_name does not describe a known comparison grid
    """

    if comparison_grid_name == 'latlon':
        raise ValueError('latlon is not a projection grid.')
    elif comparison_grid_name in ['antarctic', 'antarctic_extended']:
        projection = pyproj.Proj(
            '+proj=stere +lat_ts=-71.0 +lat_0=-90 +lon_0=0.0  +k_0=1.0 '
            '+x_0=0.0 +y_0=0.0 +ellps=WGS84')
    elif comparison_grid_name in ['arctic', 'arctic_extended']:
        projection = pyproj.Proj(
            '+proj=stere +lat_ts=75.0 +lat_0=90 +lon_0=0.0  +k_0=1.0 '
            '+x_0=0.0 +y_0=0.0 +ellps=WGS84')
    elif comparison_grid_name == 'north_atlantic':
        projection = pyproj.Proj('+proj=lcc +lon_0=-45 +lat_0=45 +lat_1=39 '
                                 '+lat_2=51 +x_0=0.0 +y_0=0.0 +ellps=WGS84')
    elif comparison_grid_name == 'north_pacific':
        projection = pyproj.Proj('+proj=lcc +lon_0=180 +lat_0=40 +lat_1=34 '
                                 '+lat_2=46 +x_0=0.0 +y_0=0.0 +ellps=WGS84')
    elif comparison_grid_name == 'subpolar_north_atlantic':
        projection = pyproj.Proj('+proj=lcc +lon_0=-40 +lat_0=54 +lat_1=40 '
                                 '+lat_2=68 +x_0=0.0 +y_0=0.0 +ellps=WGS84')
    else:
        raise ValueError(f'We missed one of the known comparison grids: '
                         f'{comparison_grid_name}')

    return projection


# A lot of duplication from MPAS-Analysis for now.
def _make_analysis_projection_map(config, mesh_name, projection_name, cores,
                                  logger):
    mesh_filename = 'restart.nc'
    section = 'files_for_e3sm'

    option_suffixes = {'antarctic': 'AntarcticStereo',
                       'arctic': 'ArcticStereo',
                       'antarctic_extended': 'AntarcticExtended',
                       'arctic_extended': 'ArcticExtended',
                       'north_atlantic': 'NorthAtlantic',
                       'north_pacific': 'NorthPacific',
                       'subpolar_north_atlantic': 'SubpolarNorthAtlantic'}

    grid_suffixes = {'antarctic': 'Antarctic_stereo',
                     'arctic': 'Arctic_stereo',
                     'antarctic_extended': 'Antarctic_stereo',
                     'arctic_extended': 'Arctic_stereo',
                     'north_atlantic': 'North_Atlantic',
                     'north_pacific': 'North_Pacific',
                     'subpolar_north_atlantic': 'Subpolar_North_Atlantic'}

    projection = _get_pyproj_projection(projection_name)
    option_suffix = option_suffixes[projection_name]
    grid_suffix = grid_suffixes[projection_name]

    in_descriptor = MpasMeshDescriptor(mesh_filename, mesh_name)

    width = config.getfloat(
        section, f'comparison{option_suffix}Width')
    option = f'comparison{option_suffix}Height'
    if config.has_option(section, option):
        height = config.getfloat(section, option)
    else:
        height = width
    res = config.getfloat(
        section, f'comparison{option_suffix}Resolution')

    xmax = 0.5 * width * 1e3
    nx = int(width / res) + 1
    x = numpy.linspace(-xmax, xmax, nx)

    ymax = 0.5 * height * 1e3
    ny = int(height / res) + 1
    y = numpy.linspace(-ymax, ymax, ny)

    out_grid_name = f'{width}x{height}km_{res}km_{grid_suffix}'
    out_descriptor = ProjectionGridDescriptor.create(projection, x, y,
                                                     mesh_name)

    _make_mapping_file(mesh_name, out_grid_name, in_descriptor, out_descriptor,
                       cores, config, logger)


def _make_mapping_file(mesh_name, out_grid_name, in_descriptor, out_descriptor,
                       cores, config, logger):

    parallel_executable = config.get('parallel', 'parallel_executable')

    mapping_file_name = f'map_{mesh_name}_to_{out_grid_name}_bilinear.nc'

    remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

    remapper.build_mapping_file(method='bilinear', mpiTasks=cores, tempdir='.',
                                logger=logger,
                                esmf_parallel_exec=parallel_executable)


def _make_moc_masks(mesh_short_name, logger, cores):
    gf = GeometricFeatures()

    mesh_filename = 'restart.nc'

    function, prefix, date = get_aggregator_by_name('MOC Basins')
    fc_mask = function(gf)

    suffix = f'{prefix}{date}'

    geojson_filename = f'{suffix}.geojson'
    mask_filename = f'{mesh_short_name}_{suffix}.nc'

    fc_mask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_region_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'cell',
            '--process_count', f'{cores}',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    mask_and_transect_filename = \
        f'{mesh_short_name}_mocBasinsAndTransects{date}.nc'

    ds_mesh = xarray.open_dataset(mesh_filename)
    ds_mask = xarray.open_dataset(mask_filename)

    ds_masks_and_transects = add_moc_southern_boundary_transects(
        ds_mask, ds_mesh, logger=logger)

    write_netcdf(ds_masks_and_transects, mask_and_transect_filename,
                 char_dim_name='StrLen')

    # make links in output directories (both inputdata and diagnostics)
    output_dir = f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}'
    symlink(
        f'../../../../../diagnostics_files/{mask_and_transect_filename}',
        f'{output_dir}/{mask_and_transect_filename}')

    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink(
        f'../../../../diagnostics_files/{mask_and_transect_filename}',
        f'{output_dir}/{mask_and_transect_filename}')
