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

from compass.io import symlink, add_input_file


def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    defaults = dict(cores=18, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    add_input_file(step, filename='README', target='../README')
    add_input_file(step, filename='restart.nc',
                   target='../{}'.format(step['restart_filename']))

    # for now, we won't define any outputs because they include the mesh short
    # name, which is not known at setup time.  Currently, this is safe because
    # no other steps depend on the outputs of this one.


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step
    """
    with_ice_shelf_cavities = step['with_ice_shelf_cavities']
    cores = step['cores']

    restart_filename = 'restart.nc'

    with xarray.open_dataset(restart_filename) as ds:
        mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
        mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
        prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)

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
    _make_analysis_polar_map(config, mesh_short_name, projection='antarctic',
                             cores=cores, logger=logger)
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

    if 'ESMF' in os.environ:
        parallel_executable = config.get('parallel', 'parallel_executable')
        esmf_path = os.environ['ESMF']
    else:
        parallel_executable = None
        esmf_path = None


    mappingFileName = 'map_{}_to_{}_bilinear.nc'.format(mesh_name, outGridName)

    remapper = Remapper(inDescriptor, outDescriptor, mappingFileName)

    remapper.build_mapping_file(method='bilinear', mpiTasks=cores, tempdir='.',
                                logger=logger, esmf_path=esmf_path,
                                esmf_parallel_exec=parallel_executable)


def _make_moc_masks(mesh_short_name, logger):
    gf = GeometricFeatures()

    mesh_filename = 'restart.nc'

    mask_filename = '{}_moc_masks.nc'.format(mesh_short_name)
    mask_and_transect_filename = '{}_moc_masks_and_transects.nc'.format(
        mesh_short_name)

    geojson_filename = 'moc_basins.geojson'

    make_moc_basins_and_transects(gf, mesh_filename, mask_and_transect_filename,
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
