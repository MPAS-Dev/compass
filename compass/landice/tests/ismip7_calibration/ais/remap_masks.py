"""
Remap the ISMIP7 Antarctic masks onto a MALI mesh.
"""

import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call

from compass.landice.ismip7.mapping import build_mapping_file
from compass.landice.tests.ismip7_calibration import datasets
from compass.step import Step

#: codes written to ``ismip7ShelfRegion``
REGION_CODES = {'none': 0, 'pig': 1, 'dotson': 2}

#: below this level of agreement, the basin numbering is probably mismatched
MIN_BASIN_AGREEMENT = 80.0


class RemapMasks(Step):
    """
    A step that remaps the ISMIP7 masks onto the MALI mesh.

    The calibration aggregates modelled melt over IMBIE2 drainage basins
    (J1, J3), over bins of equal buttressing importance (J2), and over the
    Pine Island and Dotson ice shelves (J4).  Those masks are distributed on
    the ISMIP polar stereographic grid; this step puts them on the MALI mesh
    so that the aggregation happens in MALI's own discretization.

    Remapping is nearest neighbour throughout, since every field is
    categorical.

    **Two basin-numbering conventions are in play and they differ by one.**
    ISMIP7's ``basin_numbers_ismip8km_v2.nc`` is 0-based, 0-15, with basin 9
    the Eastern Amundsen and basin 14 Ronne-Filchner, which is how the
    protocol refers to them.  MALI's ``ismip6shelfMelt_basin`` is 1-based,
    1-16.  Both are written here, under distinct names, so that neither can
    be silently reinterpreted as the other -- a mistake that produces
    plausible-looking but wrong basin aggregates rather than an error.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='remap_masks')
        self.add_output_file(filename='ismip7_masks_on_mali.nc')

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip7_calibration']
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')

        self.add_input_file(
            filename=mali_mesh_file,
            target=os.path.join(base_path_mali, mali_mesh_file))

        region_mask_file = section.get('region_mask_file')
        if region_mask_file != 'None':
            self.add_input_file(
                filename=region_mask_file,
                target=os.path.join(base_path_mali, region_mask_file))

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
        region_mask_file = section.get('region_mask_file')
        ntasks = section.getint('esmf_ntasks')

        method = config.get('ismip7_calibration_masks', 'method_remap')

        logger.info(f'Loading the ISMIP7 masks from {base_path}')
        ds_masks = _load_ismip7_masks(base_path)

        # any of the mask files defines the source grid; they share it
        src_grid_file = datasets.mask_files(base_path)['basins']

        ds_remapped = _remap_to_mali(
            config, ds_masks, src_grid_file, mali_mesh_file, mali_mesh_name,
            method, ntasks, logger)

        ds_out = _to_integer_masks(ds_remapped)

        # The ISMIP6 non-local form reads gamma0 from this same input
        # stream, and its Registry default is zero, which would silently
        # zero the melt field.  The ensemble runs at a reference value that
        # the aggregation divides out again.
        reference_gamma0 = config.getfloat('ismip7_calibration_melt',
                                           'reference_gamma0')
        ds_out['ismip6shelfMelt_gamma0'] = reference_gamma0
        ds_out['ismip6shelfMelt_gamma0'].attrs = {
            'long_name': 'gamma0 for the ISMIP6 ice-shelf melting method',
            'units': 'm yr^-1',
            'note': 'a reference value only; melt is proportional to gamma0 '
                    'and the calibration scales this away'}

        if region_mask_file != 'None':
            _cross_check_basins(ds_out, region_mask_file, logger)

        ds_out.attrs['source'] = (
            f'ISMIP7 masks from {base_path} remapped onto {mali_mesh_file}')
        ds_out.attrs['remap_method'] = method

        write_netcdf(ds_out, 'ismip7_masks_on_mali.nc')
        logger.info('Wrote ismip7_masks_on_mali.nc')


def _load_ismip7_masks(base_path):
    """Load the ISMIP7 masks on the ISMIP grid, as floats for remapping."""
    files = datasets.mask_files(base_path)

    basins = xr.open_dataset(files['basins'])['basinNumber']
    bfrn = xr.open_dataset(files['bfrn'])['BFRN_bins']
    floating = xr.open_dataset(files['floating'])['mask']

    shelves = xr.open_dataset(files['shelves'])['shelf_mask']
    if 'time' in shelves.dims:
        shelves = shelves.isel(time=0)

    # restrict Pine Island to its main trunk, as the worked example does
    pig = (shelves == datasets.PIG_ID) & (shelves['x'] > datasets.PIG_X_MAX)
    dotson = shelves == datasets.DOTSON_ID
    region = xr.where(pig, REGION_CODES['pig'],
                      xr.where(dotson, REGION_CODES['dotson'],
                               REGION_CODES['none']))

    ds = xr.Dataset()
    ds['ismip7BasinNumber'] = basins.astype(float)
    ds['ismip7BFRNBin'] = bfrn.astype(float)
    ds['ismip7FloatingMask'] = floating.astype(float)
    ds['ismip7ShelfRegion'] = region.astype(float)
    return ds


def _remap_to_mali(config, ds_masks, src_grid_file, mali_mesh_file,
                   mali_mesh_name, method, ntasks, logger):
    """
    Remap the ISMIP7 masks onto the MALI mesh.

    This goes through the shared
    :py:func:`compass.landice.ismip7.mapping.build_mapping_file` and
    ``ncremap``, the same route ``ismip7_forcing`` uses, rather than through
    pyremap's ``Remapper``.  pyremap invokes ``mpirun`` directly, which
    conflicts with the Slurm allocation on machines where compass launches
    with ``srun``; ``build_mapping_file`` uses the configured
    ``parallel_executable`` instead.
    """
    res = datasets.ISMIP_RESOLUTION_KM
    mapping_file = f'map_ismip{res}km_to_{mali_mesh_name}_{method}.nc'
    source_file = 'ismip7_masks_source.nc'
    remapped_file = 'ismip7_masks_remapped.nc'

    # the masks are assembled in memory, so write them back onto the ISMIP
    # grid for ncremap; the source grid file supplies the x/y coordinates
    # that the SCRIP description needs
    with xr.open_dataset(src_grid_file) as ds_grid:
        ds_source = ds_masks.assign_coords(x=ds_grid['x'], y=ds_grid['y'])
    write_netcdf(ds_source, source_file)

    logger.info(f'Building mapping file {mapping_file}')
    build_mapping_file(config, logger, source_file, mapping_file,
                       mali_mesh_file=mali_mesh_file, method_remap=method,
                       projection='ais-bedmap2', ntasks=ntasks)

    logger.info('Remapping the masks onto the MALI mesh')
    variables = ','.join(sorted(ds_masks.data_vars))
    check_call(['ncremap', '-i', source_file, '-o', remapped_file,
                '-m', mapping_file, '-v', variables], logger=logger)

    ds_remapped = xr.load_dataset(remapped_file)
    if 'ncol' in ds_remapped.dims:
        ds_remapped = ds_remapped.rename({'ncol': 'nCells'})

    for path in (source_file, remapped_file):
        if os.path.exists(path):
            os.remove(path)

    return ds_remapped


def _to_integer_masks(ds_remapped):
    """
    Round the remapped fields to integers and derive the MALI basin field.

    Nearest-neighbour remapping should already return exact source values,
    but they come back as floats; rounding makes the intent explicit and
    guards against a different method being used.
    """
    ds = xr.Dataset()

    basin0 = ds_remapped['ismip7BasinNumber']
    valid = basin0.notnull()
    basin0 = basin0.round().fillna(-1).astype(np.int32)

    ds['ismip7BasinNumber'] = basin0
    ds['ismip7BasinNumber'].attrs = {
        'long_name': 'IMBIE2 drainage basin number, ISMIP7 convention',
        'convention': '0-based, 0-15; basin 9 is Eastern Amundsen, '
                      'basin 14 is Ronne-Filchner'}

    # MALI's melt parameterization expects the 1-based convention
    ds['ismip6shelfMelt_basin'] = \
        xr.where(valid, basin0 + 1, 0).astype(np.int32)
    ds['ismip6shelfMelt_basin'].attrs = {
        'long_name': 'basin number for the MALI melt parameterization',
        'convention': '1-based, 1-16, equal to ismip7BasinNumber + 1; '
                      '0 marks cells with no basin'}

    # MALI reads the per-basin thermal-forcing correction from the same input
    # stream as the basin numbers, so it has to be present or the run fails.
    # The calibration fits dT_b *after* the melt parameter, and uses zero
    # throughout; the fit_delta_t step writes calibrated values over this.
    ds['ismip6shelfMelt_deltaT'] = \
        xr.zeros_like(ds['ismip6shelfMelt_basin'], dtype=float)
    ds['ismip6shelfMelt_deltaT'].attrs = {
        'long_name': 'basin-wide thermal forcing correction',
        'units': 'degC',
        'note': 'zero as written here; the fit_delta_t step fits one value '
                'per basin and writes them back over this field'}

    bfrn = ds_remapped['ismip7BFRNBin']
    ds['ismip7BFRNBin'] = bfrn.round().fillna(-1).astype(np.int32)
    ds['ismip7BFRNBin'].attrs = {
        'long_name': 'buttressing flux response number bin',
        'convention': '0-9; bin 0 is passive ice, bin 9 the most '
                      'buttressing-relevant; -1 marks cells with no bin'}

    ds['ismip7FloatingMask'] = \
        ds_remapped['ismip7FloatingMask'].round().fillna(0).astype(np.int32)
    ds['ismip7FloatingMask'].attrs = {
        'long_name': 'ISMIP7 floating-ice mask',
        'convention': '1 where floating, 0 otherwise',
        'note': 'This is the observed ISMIP7 shelf extent, for diagnostics '
                'such as comparing modelled with observed shelf area.  Melt '
                'aggregation uses MALI own floating cells, since the '
                'calibration holds the model accountable for its own shelf '
                'extent.'}

    region = ds_remapped['ismip7ShelfRegion'].round().fillna(0)
    ds['ismip7ShelfRegion'] = region.astype(np.int32)
    ds['ismip7ShelfRegion'].attrs = {
        'long_name': 'ice-shelf region for calibration term J4',
        'convention': ', '.join(f'{value} = {name}'
                                for name, value in REGION_CODES.items())}
    return ds


def _cross_check_basins(ds_masks, region_mask_file, logger):
    """
    Compare the remapped basin field with MALI's existing region mask.

    The two are built from different sources -- ISMIP7 IMBIE2 v3 here, versus
    the ISMIP6 regions rasterized with ``geometric_features`` in 2022 -- so
    exact agreement is not expected.  What matters is that the *offset* is
    the expected one: an off-by-one would show near-zero agreement
    everywhere, whereas genuine boundary differences show up in individual
    basins.

    Agreement is reported in **both directions**, because the two masks do
    not cover the same cells and a one-directional figure is misleading.  A
    basin that ISMIP7 draws smaller than ISMIP6 scores high conditioned on
    ours and low conditioned on theirs; that is a real difference in the
    basin outlines, not an error.

    Raises
    ------
    ValueError
        If overall agreement is below :py:data:`MIN_BASIN_AGREEMENT`, which
        means the numbering conventions are almost certainly mismatched
    """
    ds_region = xr.open_dataset(region_mask_file)
    if 'regionCellMasks' not in ds_region:
        logger.warning('No regionCellMasks in the region mask file; skipping '
                       'the basin cross-check')
        return

    masks = ds_region['regionCellMasks'].values
    # column index + 1 is MALI's 1-based basin number
    mali_basin = np.zeros(masks.shape[0], dtype=np.int32)
    for col in range(masks.shape[1]):
        mali_basin[masks[:, col] == 1] = col + 1

    ours = ds_masks['ismip6shelfMelt_basin'].values
    floating = ds_masks['ismip7FloatingMask'].values == 1
    both = (ours > 0) & (mali_basin > 0)
    if not both.any():
        logger.warning('No cells carry both basin numbers; skipping the '
                       'basin cross-check')
        return

    logger.info('')
    logger.info('Cross-check against the existing MALI region mask:')
    for label, sel in (('all cells', both),
                       ('floating only', both & floating)):
        if not sel.any():
            continue
        frac = 100.0 * (ours[sel] == mali_basin[sel]).mean()
        logger.info(f'  {label:14s} agreement {frac:5.1f}%  '
                    f'(n={int(sel.sum())})')

    overall = 100.0 * (ours[both] == mali_basin[both]).mean()

    logger.info('  per basin, conditioned on each mask in turn:')
    logger.info(f'    {"basin":>5s} {"ours->theirs":>13s} {"n":>7s} '
                f'{"theirs->ours":>13s} {"n":>7s}')
    for basin in range(1, masks.shape[1] + 1):
        sel_ours = both & (ours == basin)
        sel_theirs = both & (mali_basin == basin)
        if not (sel_ours.any() or sel_theirs.any()):
            continue
        forward = (100.0 * (mali_basin[sel_ours] == basin).mean()
                   if sel_ours.any() else float('nan'))
        backward = (100.0 * (ours[sel_theirs] == basin).mean()
                    if sel_theirs.any() else float('nan'))
        logger.info(f'    {basin:5d} {forward:12.1f}% '
                    f'{int(sel_ours.sum()):7d} {backward:12.1f}% '
                    f'{int(sel_theirs.sum()):7d}')
    logger.info('  (a basin drawn smaller by ISMIP7 than by ISMIP6 scores '
                'high in the first column and low in the second; that is a '
                'real difference in the outlines, not an error)')
    logger.info('')

    if overall < MIN_BASIN_AGREEMENT:
        raise ValueError(
            f'The remapped ISMIP7 basins agree with the existing MALI region '
            f'mask for only {overall:.1f}% of cells, below the '
            f'{MIN_BASIN_AGREEMENT:.0f}% threshold.  The basin numbering '
            f'conventions are probably mismatched: ISMIP7 is 0-based and '
            f'MALI is 1-based, so MALI basin 10 is ISMIP7 basin 9.  Using '
            f'one where the other is expected mis-assigns every basin and '
            f'still produces plausible-looking numbers.')
