import glob
import os

import numpy
import pyproj
from pyremap import (
    MpasCellMeshDescriptor,
    MpasVertexMeshDescriptor,
    ProjectionGridDescriptor,
    Remapper,
    get_lat_lon_descriptor,
)

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class DiagnosticMaps(FilesForE3SMStep):
    """
    A step for creating mapping files for use in MPAS-Analysis
    """

    def __init__(self, test_case):
        """
        Create a step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='diagnostics_maps', ntasks=36,
                         min_tasks=1)

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        make_diagnostics_maps(self.config, self.logger, self.mesh_short_name,
                              self.ntasks)


def make_diagnostics_maps(config, logger, mesh_short_name, ntasks):
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

    ntasks : int
        The number of cores to use to build mapping files
    """
    link_dir = '../assembled_files/diagnostics/mpas_analysis/maps'

    try:
        os.makedirs(link_dir)
    except OSError:
        pass

    _make_analysis_lat_lon_map(config, mesh_short_name, ntasks, logger)
    for projection_name in ['antarctic', 'arctic', 'antarctic_extended',
                            'arctic_extended', 'north_atlantic',
                            'north_pacific', 'subpolar_north_atlantic']:
        _make_analysis_projection_map(config, mesh_short_name, projection_name,
                                      ntasks, logger)

    # make links in output directory
    files = glob.glob('map_*')

    # make links in output directory
    for filename in files:
        symlink(os.path.abspath(filename),
                f'{link_dir}/{filename}')


def _make_analysis_lat_lon_map(config, mesh_name, ntasks, logger):
    mesh_filename = 'restart.nc'

    lat_res = config.getfloat('files_for_e3sm', 'comparisonLatResolution')
    lon_res = config.getfloat('files_for_e3sm', 'comparisonLonResolution')

    # modify the resolution of the global lat-lon grid as desired
    out_descriptor = get_lat_lon_descriptor(dLon=lat_res,
                                            dLat=lon_res)
    out_grid_name = out_descriptor.meshName

    _make_mapping_file(mesh_name, out_grid_name, mesh_filename, out_descriptor,
                       ntasks, config, logger)


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
def _make_analysis_projection_map(config, mesh_name, projection_name, ntasks,
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

    _make_mapping_file(mesh_name, out_grid_name, mesh_filename, out_descriptor,
                       ntasks, config, logger)


def _make_mapping_file(mesh_name, out_grid_name, mesh_filename, out_descriptor,
                       ntasks, config, logger):

    parallel_executable = config.get('parallel', 'parallel_executable')

    in_descriptor = MpasCellMeshDescriptor(mesh_filename, mesh_name)

    mapping_file_name = f'map_{mesh_name}_to_{out_grid_name}_bilinear.nc'

    remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

    remapper.build_mapping_file(method='bilinear', mpiTasks=ntasks,
                                tempdir='.', logger=logger,
                                esmf_parallel_exec=parallel_executable)

    # now the same on vertices (e.g. for streamfunctions)
    in_descriptor = MpasVertexMeshDescriptor(mesh_filename, mesh_name)
    mapping_file_name = \
        f'map_{mesh_name}_vertices_to_{out_grid_name}_bilinear.nc'

    remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

    remapper.build_mapping_file(method='bilinear', mpiTasks=ntasks,
                                tempdir='.', logger=logger,
                                esmf_parallel_exec=parallel_executable)
