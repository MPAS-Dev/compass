import os
import xarray
import glob

from pyremap import get_lat_lon_descriptor, get_polar_descriptor, \
    MpasMeshDescriptor, Remapper
from geometric_features import GeometricFeatures
from geometric_features.aggregation import get_aggregator_by_name
from mpas_tools.mesh.conversion import mask
from mpas_tools.io import write_netcdf
from mpas_tools.ocean.moc import make_moc_basins_and_transects

from compass.io import symlink
from compass.step import Step


class DiagnosticsFiles(Step):
    """
    A step for creating files needed for the Meridional Overturning Circulation
     analysis member and diagnostics from MPAS-Analysis

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

        super().__init__(test_case, name='diagnostics_files', cores=18,
                         min_cores=1, threads=1)

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
        with_ice_shelf_cavities = self.with_ice_shelf_cavities
        cores = self.cores
        config = self.config
        logger = self.logger

        restart_filename = 'restart.nc'

        with xarray.open_dataset(restart_filename) as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']

        for directory in [
                '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
                    mesh_short_name),
                '../assembled_files/diagnostics/mpas_analysis/region_masks',
                '../assembled_files/diagnostics/mpas_analysis/maps']:
            try:
                os.makedirs(directory)
            except OSError:
                pass

        _make_moc_masks(mesh_short_name, logger)

        gf = GeometricFeatures()

        region_groups = ['Antarctic Regions', 'Arctic Ocean Regions',
                         'Arctic Sea Ice Regions', 'Ocean Basins',
                         'Ocean Subbasins', 'ISMIP6 Regions',
                         'Transport Transects']

        if with_ice_shelf_cavities:
            region_groups.append('Ice Shelves')

        for region_group in region_groups:
            function, prefix, date = get_aggregator_by_name(region_group)

            suffix = '{}{}'.format(prefix, date)

            fcMask = function(gf)
            _make_region_masks(mesh_short_name, suffix=suffix, fcMask=fcMask,
                               logger=logger)

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


def _make_region_masks(mesh_name, suffix, fcMask, logger):
    mesh_filename = 'restart.nc'

    geojson_filename = '{}.geojson'.format(suffix)
    mask_filename = '{}_{}.nc'.format(mesh_name, suffix)

    fcMask.to_geojson(geojson_filename)

    dsMesh = xarray.open_dataset(mesh_filename)

    dsMask = mask(dsMesh, fcMask=fcMask, logger=logger)

    write_netcdf(dsMask, mask_filename)

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


def _make_moc_masks(mesh_short_name, logger):
    gf = GeometricFeatures()

    mesh_filename = 'restart.nc'

    mask_filename = '{}_moc_masks.nc'.format(mesh_short_name)
    mask_and_transect_filename = '{}_moc_masks_and_transects.nc'.format(
        mesh_short_name)

    geojson_filename = 'moc_basins.geojson'

    make_moc_basins_and_transects(gf, mesh_filename,
                                  mask_and_transect_filename,
                                  geojson_filename=geojson_filename,
                                  mask_filename=mask_filename,
                                  logger=logger)

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
