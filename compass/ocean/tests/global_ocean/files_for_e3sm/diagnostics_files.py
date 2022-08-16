import os
import xarray
import glob

from pyremap import get_lat_lon_descriptor, get_polar_descriptor, \
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
                            target='../{}'.format(restart_filename))

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
            '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
                mesh_short_name),
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
        suffix = '{}{}'.format(prefix, date)
        fcMask = function(gf)
        _make_region_masks(mesh_short_name, suffix=suffix, fcMask=fcMask,
                           logger=logger, cores=cores)

    transect_groups = ['Transport Transects']
    for transect_group in transect_groups:
        function, prefix, date = get_aggregator_by_name(transect_group)
        suffix = '{}{}'.format(prefix, date)
        fcMask = function(gf)
        _make_transect_masks(mesh_short_name, suffix=suffix, fcMask=fcMask,
                             logger=logger, cores=cores)

    _make_analysis_lat_lon_map(config, mesh_short_name, cores, logger)
    _make_analysis_polar_map(config, mesh_short_name,
                             projection='antarctic', cores=cores,
                             logger=logger)
    _make_analysis_polar_map(config, mesh_short_name, projection='arctic',
                             cores=cores, logger=logger)

    # make links in output directory
    files = glob.glob('map_*')

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/maps'
    for filename in files:
        symlink('../../../../diagnostics_files/{}'.format(filename),
                '{}/{}'.format(output_dir, filename))


def _make_region_masks(mesh_name, suffix, fcMask, logger, cores):
    mesh_filename = 'restart.nc'

    geojson_filename = '{}.geojson'.format(suffix)
    mask_filename = '{}_{}.nc'.format(mesh_name, suffix)

    fcMask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_region_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'cell',
            '--process_count', '{}'.format(cores),
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink('../../../../diagnostics_files/{}'.format(mask_filename),
            '{}/{}'.format(output_dir, mask_filename))


def _make_transect_masks(mesh_name, suffix, fcMask, logger, cores,
                         subdivision_threshold=10e3):
    mesh_filename = 'restart.nc'

    geojson_filename = '{}.geojson'.format(suffix)
    mask_filename = '{}_{}.nc'.format(mesh_name, suffix)

    fcMask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_transect_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'edge',
            '-s', '{}'.format(subdivision_threshold),
            '--process_count', '{}'.format(cores),
            '--add_edge_sign',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    # make links in output directory
    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink('../../../../diagnostics_files/{}'.format(mask_filename),
            '{}/{}'.format(output_dir, mask_filename))


def _make_analysis_lat_lon_map(config, mesh_name, cores, logger):
    mesh_filename = 'restart.nc'

    inDescriptor = MpasMeshDescriptor(mesh_filename, mesh_name)

    comparisonLatResolution = config.getfloat('files_for_e3sm',
                                              'comparisonLatResolution')
    comparisonLonResolution = config.getfloat('files_for_e3sm',
                                              'comparisonLonResolution')

    # modify the resolution of the global lat-lon grid as desired
    outDescriptor = get_lat_lon_descriptor(dLon=comparisonLatResolution,
                                           dLat=comparisonLonResolution)
    outGridName = outDescriptor.meshName

    _make_mapping_file(mesh_name, outGridName, inDescriptor, outDescriptor,
                       cores, config, logger)


def _make_analysis_polar_map(config, mesh_name, projection, cores, logger):
    mesh_filename = 'restart.nc'

    upperProj = projection[0].upper() + projection[1:]

    inDescriptor = MpasMeshDescriptor(mesh_filename, mesh_name)

    comparisonStereoWidth = config.getfloat(
        'files_for_e3sm', 'comparison{}StereoWidth'.format(upperProj))
    comparisonStereoResolution = config.getfloat(
        'files_for_e3sm', 'comparison{}StereoResolution'.format(upperProj))

    outDescriptor = get_polar_descriptor(Lx=comparisonStereoWidth,
                                         Ly=comparisonStereoWidth,
                                         dx=comparisonStereoResolution,
                                         dy=comparisonStereoResolution,
                                         projection=projection)

    outGridName = '{}x{}km_{}km_{}_stereo'.format(
        comparisonStereoWidth,  comparisonStereoWidth,
        comparisonStereoResolution, upperProj)

    _make_mapping_file(mesh_name, outGridName, inDescriptor, outDescriptor,
                       cores, config, logger)


def _make_mapping_file(mesh_name, outGridName, inDescriptor, outDescriptor,
                       cores, config, logger):

    parallel_executable = config.get('parallel', 'parallel_executable')

    mappingFileName = 'map_{}_to_{}_bilinear.nc'.format(mesh_name, outGridName)

    remapper = Remapper(inDescriptor, outDescriptor, mappingFileName)

    remapper.build_mapping_file(method='bilinear', mpiTasks=cores, tempdir='.',
                                logger=logger,
                                esmf_parallel_exec=parallel_executable)


def _make_moc_masks(mesh_short_name, logger, cores):
    gf = GeometricFeatures()

    mesh_filename = 'restart.nc'

    function, prefix, date = get_aggregator_by_name('MOC Basins')
    fcMask = function(gf)

    suffix = '{}{}'.format(prefix, date)

    geojson_filename = '{}.geojson'.format(suffix)
    mask_filename = '{}_{}.nc'.format(mesh_short_name, suffix)

    fcMask.to_geojson(geojson_filename)

    # these defaults may have been updated from config options -- pass them
    # along to the subprocess
    netcdf_format = mpas_tools.io.default_format
    netcdf_engine = mpas_tools.io.default_engine

    args = ['compute_mpas_region_masks',
            '-m', mesh_filename,
            '-g', geojson_filename,
            '-o', mask_filename,
            '-t', 'cell',
            '--process_count', '{}'.format(cores),
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    mask_and_transect_filename = '{}_mocBasinsAndTransects{}.nc'.format(
        mesh_short_name, date)

    dsMesh = xarray.open_dataset(mesh_filename)
    dsMask = xarray.open_dataset(mask_filename)

    dsMasksAndTransects = add_moc_southern_boundary_transects(
        dsMask, dsMesh, logger=logger)

    write_netcdf(dsMasksAndTransects, mask_and_transect_filename,
                 char_dim_name='StrLen')

    # make links in output directories (both inputdata and diagnostics)
    output_dir = '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
        mesh_short_name)
    symlink(
        '../../../../../diagnostics_files/{}'.format(
            mask_and_transect_filename),
        '{}/{}'.format(output_dir, mask_and_transect_filename))

    output_dir = '../assembled_files/diagnostics/mpas_analysis/' \
                 'region_masks'
    symlink(
        '../../../../diagnostics_files/{}'.format(
            mask_and_transect_filename),
        '{}/{}'.format(output_dir, mask_and_transect_filename))
