import os
import xarray as xr

from geometric_features import GeometricFeatures
from geometric_features.aggregation import get_aggregator_by_name
from mpas_tools.logging import check_call
from mpas_tools.ocean.moc import add_moc_southern_boundary_transects
from mpas_tools.io import write_netcdf
import mpas_tools.io

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step \
    import FilesForE3SMStep


class DiagnosticMasks(FilesForE3SMStep):
    """
    A step for creating region masks needed for the Meridional Overturning
    Circulation analysis member and diagnostics from MPAS-Analysis
    """

    def __init__(self, test_case):
        """
        Create a step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """

        super().__init__(test_case, name='diagnostics_masks', cpus_per_task=18,
                         min_cpus_per_task=1)

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        make_diagnostics_files(self.logger, self.mesh_short_name,
                               self.with_ice_shelf_cavities,
                               self.cpus_per_task)


def make_diagnostics_files(logger, mesh_short_name, with_ice_shelf_cavities,
                           cpus_per_task):
    """
    Run this step of the testcase

    Parameters
    ----------
    logger : logging.Logger
        A logger for output from the step

    mesh_short_name : str
        The E3SM short name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh has ice-shelf cavities

    cpus_per_task : int
        The number of cores to use to build masks
    """

    mask_dir = '../assembled_files/diagnostics/mpas_analysis/region_masks'
    try:
        os.makedirs(mask_dir)
    except FileExistsError:
        pass

    ocean_inputdata_dir = \
        f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}'
    moc_mask_dirs = [mask_dir, ocean_inputdata_dir]

    _make_moc_masks(mesh_short_name, logger, cpus_per_task, moc_mask_dirs)

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
                           logger=logger, cpus_per_task=cpus_per_task,
                           output_dir=mask_dir)

    transect_groups = ['Transport Transects']
    for transect_group in transect_groups:
        function, prefix, date = get_aggregator_by_name(transect_group)
        suffix = f'{prefix}{date}'
        fc_mask = function(gf)
        _make_transect_masks(mesh_short_name, suffix=suffix, fc_mask=fc_mask,
                             logger=logger, cpus_per_task=cpus_per_task,
                             output_dir=mask_dir)


def _make_region_masks(mesh_name, suffix, fc_mask, logger, cpus_per_task,
                       output_dir):
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
            '--process_count', f'{cpus_per_task}',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    # make links in output directory
    symlink(os.path.abspath(mask_filename),
            f'{output_dir}/{mask_filename}')


def _make_transect_masks(mesh_name, suffix, fc_mask, logger, cpus_per_task,
                         output_dir, subdivision_threshold=10e3):
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
            '--process_count', f'{cpus_per_task}',
            '--add_edge_sign',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    symlink(os.path.abspath(mask_filename),
            f'{output_dir}/{mask_filename}')


def _make_moc_masks(mesh_short_name, logger, cpus_per_task, moc_mask_dirs):
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
            '--process_count', f'{cpus_per_task}',
            '--format', netcdf_format,
            '--engine', netcdf_engine]
    check_call(args, logger=logger)

    mask_and_transect_filename = \
        f'{mesh_short_name}_mocBasinsAndTransects{date}.nc'

    ds_mesh = xr.open_dataset(mesh_filename)
    ds_mask = xr.open_dataset(mask_filename)

    ds_masks_and_transects = add_moc_southern_boundary_transects(
        ds_mask, ds_mesh, logger=logger)

    write_netcdf(ds_masks_and_transects, mask_and_transect_filename,
                 char_dim_name='StrLen')

    # make links in output directories (both inputdata and diagnostics)
    for output_dir in moc_mask_dirs:
        symlink(os.path.abspath(mask_and_transect_filename),
                f'{output_dir}/{mask_and_transect_filename}')
