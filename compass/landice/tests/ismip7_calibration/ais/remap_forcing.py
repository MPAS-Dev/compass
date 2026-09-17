"""
Remap the ISMIP7 calibration thermal forcing onto a MALI mesh.
"""

import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.ismip7.mapping import build_mapping_file
from compass.landice.ismip7.remap import extrapolate_source
from compass.landice.tests.ismip7_calibration import datasets
from compass.step import Step


class RemapForcing(Step):
    """
    A step that remaps the ISMIP7 calibration thermal forcing onto the MALI
    mesh, one file per ocean state.

    This mirrors ``ismip7_forcing/ocean_thermal``, driven by the calibration
    ocean states rather than by CMIP scenario files.

    **Only thermal forcing is remapped.**  The ISMIP7 ocean forcing processed
    for the MALI projections carries thermal forcing and no salinity, so a
    projection has no salinity field to read.  The melt parameterization is
    run with a constant salinity instead
    (``config_ismip7_melt_salinity_source = 'constant'``); calibrating
    against a spatially varying salinity would tune the melt parameter for
    physics the projections cannot run.

    Attributes
    ----------
    states : list of compass.landice.tests.ismip7_calibration.datasets.OceanState
        The ocean states to remap
    """  # noqa: E501

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='remap_forcing')
        self.states = []

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip7_calibration']
        base_path = section.get('base_path_ismip7')
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')
        subset = section.get('ocean_state_subset')

        self.add_input_file(
            filename=mali_mesh_file,
            target=os.path.join(base_path_mali, mali_mesh_file))

        self.states = datasets.ocean_states(base_path, subset=subset)
        missing = datasets.missing_files(self.states)
        # only the thermal forcing is needed; a missing salinity file is not
        # a problem, since a constant salinity is used
        missing = [(name, path) for name, path in missing
                   if not path.endswith('_S.nc') and '/so/' not in path]
        if missing:
            listing = '\n  '.join(f'{name}: {path}' for name, path in missing)
            raise FileNotFoundError(
                f'{len(missing)} ISMIP7 thermal-forcing files are '
                f'missing:\n  {listing}')

        for state in self.states:
            self.add_output_file(filename=f'forcing_{state.name}.nc')

        self.ntasks = section.getint('esmf_ntasks')
        self.min_tasks = self.ntasks

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        section = config['ismip7_calibration']
        base_path = section.get('base_path_ismip7')
        mali_mesh_file = section.get('mali_mesh_file')
        mali_mesh_name = section.get('mali_mesh_name')
        ntasks = section.getint('esmf_ntasks')

        method = config.get('ismip7_calibration_forcing', 'method_remap')

        res = datasets.ISMIP_RESOLUTION_KM
        mapping_file = (f'map_ismip{res}km_to_{mali_mesh_name}_'
                        f'{method}.nc')
        # any calibration forcing file defines the source grid; they share it
        build_mapping_file(config, logger, self.states[0].tf_file,
                           mapping_file, mali_mesh_file=mali_mesh_file,
                           method_remap=method, projection='ais-bedmap2',
                           ntasks=ntasks)

        clim_tf_file, _ = datasets.climatology_files(base_path)
        clim_tf = xr.open_dataset(clim_tf_file)['tf']

        for state in self.states:
            output_file = f'forcing_{state.name}.nc'
            if os.path.exists(output_file):
                logger.info(f'  {output_file} exists, skipping')
                continue
            logger.info(f'Remapping {state.name}')
            _remap_state(state, clim_tf, mapping_file, output_file, logger)


def _remap_state(state, clim_tf, mapping_file, output_file, logger):
    """Fill, extrapolate, remap and rename one ocean state."""
    filled_file = f'filled_{state.name}.nc'
    extrap_file = f'extrap_{state.name}.nc'
    remapped_file = f'remapped_{state.name}.nc'

    filled = _fill_from_climatology(state, clim_tf, filled_file, logger)

    # a safety net: after the climatology fill there should be nothing left
    # to extrapolate, but a stray gap would otherwise be blended into
    # neighbouring cells by the bilinear remap
    extrapolate_source(filled_file, extrap_file, 'tf', logger)

    check_call(['ncremap', '-i', extrap_file, '-o', remapped_file,
                '-m', mapping_file, '-v', 'tf'], logger=logger)

    _to_mali_form(remapped_file, state, output_file, filled)

    for path in (filled_file, extrap_file, remapped_file):
        if os.path.exists(path):
            os.remove(path)


def _fill_from_climatology(state, clim_tf, filled_file, logger):
    """
    Fill gaps in a partially covering ocean state from the climatology.

    The ISMIP7 near-ice-shelf observational datasets apply a single profile
    to the Amundsen basin only (protocol Sect. A10) and are undefined
    elsewhere; about 6% of the ISMIP grid is valid.  The regional *model*
    datasets, by contrast, are distributed already filled with the ISMIP7
    climatology outside their domains (protocol Sect. A9), so they arrive
    complete.

    MALI needs valid forcing everywhere it has ice, so the observational
    states are filled the same way the model states already were.  This does
    not affect the calibration: J4 aggregates only over Pine Island and
    Dotson, both inside the covered basin.

    Filling happens *before* remapping, so that interpolation never blends a
    real value with a missing one at the edge of the covered region.

    Returns
    -------
    fraction : float
        Fraction of the source grid that was filled, recorded in the output
    """
    ds = xr.open_dataset(state.tf_file, decode_times=False)
    valid = ds['tf'].notnull()
    fraction = float((~valid).mean())

    if fraction > 0.0:
        logger.info(f'  filling {100.0 * fraction:.1f}% of the source grid '
                    f'from the climatology')
        attrs = ds['tf'].attrs
        ds['tf'] = ds['tf'].where(valid, clim_tf)
        ds['tf'].attrs = attrs

    if '_FillValue' in ds['tf'].encoding:
        del ds['tf'].encoding['_FillValue']
    write_netcdf(ds, filled_file)
    ds.close()
    return fraction


def _to_mali_form(remapped_file, state, output_file, filled):
    """
    Put the remapped thermal forcing into the names, dimensions and order
    MALI expects.

    Two traps are avoided here.  MALI takes dimension sizes from its input
    stream, so ``nISMIP6OceanLayers`` has to be a real dimension of this
    file.  And a variable whose name matches a dimension breaks MALI's
    reader, so the vertical coordinate is written as
    ``ismip6shelfMelt_zOcean`` rather than as a coordinate variable named
    after the dimension.
    """
    ds = xr.open_dataset(remapped_file, decode_times=False)

    z_ocean = ds['z'].values

    rename_dims = {}
    if 'ncol' in ds.dims:
        rename_dims['ncol'] = 'nCells'
    if 'z' in ds.dims:
        rename_dims['z'] = 'nISMIP6OceanLayers'
    ds = ds.rename(rename_dims)
    ds = ds.rename({'tf': 'ismip6shelfMelt_3dThermalForcing'})

    tf = ds['ismip6shelfMelt_3dThermalForcing']
    if 'Time' not in tf.dims:
        tf = tf.expand_dims('Time', axis=0)
    # Registry order is ``nISMIP6OceanLayers nCells Time``, which in C order
    # is Time, nCells, nISMIP6OceanLayers
    tf = tf.transpose('Time', 'nCells', 'nISMIP6OceanLayers')

    ds_out = xr.Dataset()
    ds_out['ismip6shelfMelt_3dThermalForcing'] = tf.astype(float)
    ds_out['ismip6shelfMelt_3dThermalForcing'].attrs = {
        'long_name': 'thermal forcing for the ISMIP6/ISMIP7 ice-shelf '
                     'melting methods',
        'units': 'degC'}
    ds_out['ismip6shelfMelt_3dThermalForcing'].encoding.clear()

    ds_out['ismip6shelfMelt_zOcean'] = ('nISMIP6OceanLayers', z_ocean)
    ds_out['ismip6shelfMelt_zOcean'].attrs = {
        'long_name': 'depth coordinate for the ocean thermal forcing',
        'units': 'm'}

    ds_out['xtime'] = ('Time', ['0000-01-01_00:00:00'.ljust(64)])
    ds_out['xtime'] = ds_out.xtime.astype('S')

    ds_out.attrs['ocean_state'] = state.name
    ds_out.attrs['objective_term'] = state.term
    ds_out.attrs['source_file'] = state.tf_file
    ds_out.attrs['climatology_filled_fraction'] = filled
    if state.basins is not None:
        ds_out.attrs['constrains_ismip7_basins'] = ', '.join(
            str(basin) for basin in state.basins)

    finite = np.isfinite(ds_out['ismip6shelfMelt_3dThermalForcing'].values)
    if not finite.all():
        raise ValueError(
            f'The remapped thermal forcing for {state.name} is not finite '
            f'everywhere: {int((~finite).sum())} of {finite.size} values are '
            f'missing.  MALI would produce invalid melt in those cells.')

    # Time must be UNLIMITED or MALI's reader mishandles the file, so this
    # is written with xarray directly rather than through write_netcdf
    ds_out.to_netcdf(output_file, unlimited_dims=['Time'])
    ds.close()
