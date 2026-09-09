"""
Aggregate the MALI melt ensemble into the four objective-function terms.
"""

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration.ais import melt_model
from compass.landice.tests.ismip7_calibration.terms import (
    average_by_group,
    integrate_by_group,
)
from compass.step import Step

#: cell dimension of an MPAS mesh
CELL_DIMS = ('nCells',)


class Aggregate(Step):
    """
    A step that turns the MALI melt fields into unit-parameter aggregates of
    the four objective-function terms.

    All aggregation is **area-weighted**.  On a 4-20 km variable-resolution
    mesh a plain mean is wrong: integrals use ``melt * areaCell`` and means
    are weighted by ``areaCell``.

    Because melt is exactly proportional to the melt parameter, each
    aggregate is divided by the reference parameter the ensemble was run at,
    giving a unit aggregate that the calibration scales onto the whole
    parameter grid.  That scaling is exact, not an approximation.

    This step also reports MALI's per-basin ice-shelf area against the
    observed ISMIP7 extent.  J1, J2 and J4 are integrals, so any shelf-area
    mismatch enters the calibrated parameter directly; the protocol intends
    that, but it should be reported alongside the result.

    Attributes
    ----------
    melt_forms : list of str
        The melt forms to aggregate

    states : list of compass.landice.tests.ismip7_calibration.datasets.OceanState
        The ocean states in the ensemble
    """  # noqa: E501

    def __init__(self, test_case, melt_forms, states):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        melt_forms : list of str
            The melt forms to aggregate

        states : list of datasets.OceanState
            The ocean states in the ensemble
        """
        super().__init__(test_case=test_case, name='aggregate')
        self.melt_forms = melt_forms
        self.states = states

        self.add_input_file(filename='masks.nc',
                            target='../remap_masks/ismip7_masks_on_mali.nc')
        for melt_form in melt_forms:
            for state in states:
                self.add_input_file(
                    filename=f'melt_{melt_form}_{state.name}.nc',
                    target=f'../{melt_form}_{state.name}/output_melt.nc')
            self.add_output_file(filename=f'aggregates_{melt_form}.nc')
        self.add_output_file(filename='shelf_area.nc')

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip7_calibration']
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')
        self.add_input_file(
            filename='mesh.nc',
            target=f'{base_path_mali}/{mali_mesh_file}')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['ismip7_calibration_melt']
        reference = {'ismip7': section.getfloat('reference_k'),
                     'ismip6': section.getfloat('reference_gamma0')}

        static = _load_static(logger)

        for melt_form in self.melt_forms:
            logger.info(f'Aggregating the {melt_form} ensemble')
            ds = _aggregate_form(self.states, melt_form,
                                 reference[melt_form], static, logger)
            write_netcdf(ds, f'aggregates_{melt_form}.nc')

        _report_shelf_area(static, logger)


def _load_static(logger):
    """Load the mesh, masks and MALI's own floating-cell mask."""
    ds_mesh = xr.open_dataset('mesh.nc')
    ds_masks = xr.open_dataset('masks.nc')

    area = ds_mesh['areaCell']

    basins = ds_masks['ismip7BasinNumber']
    bfrn = ds_masks['ismip7BFRNBin']
    region_code = ds_masks['ismip7ShelfRegion']
    observed_floating = ds_masks['ismip7FloatingMask'] == 1

    # map the region codes onto the labels the targets use
    region_label = xr.where(region_code == 1, 'pig',
                            xr.where(region_code == 2, 'dotson', ''))

    logger.info(f'  {int((basins >= 0).sum())} cells carry a basin number')
    return dict(area=area, basins=basins.where(basins >= 0),
                bfrn=bfrn.where(bfrn >= 0), region_label=region_label,
                observed_floating=observed_floating)


def _melt_from_run(filename, reference):
    """
    Read one melt field, in kg m-2 yr-1 per unit parameter.

    Dividing by the reference parameter the run used gives melt at a unit
    parameter, which is exact because melt is proportional to the parameter.

    The contributing cells are the ones MALI actually computed melt for --
    floating *and* connected to the open ocean.  Including the rest would
    not change an integral, since their melt is zero, but it would dilute
    the area-weighted basin means that J3 is built from.
    """
    fields = melt_model.read_run(filename)
    return fields['melt'] / reference, fields['floating']


def _aggregate_form(states, melt_form, reference, static, logger):
    """Build the four unit aggregates for one melt form."""
    area = static['area']
    basins, bfrn = static['basins'], static['bfrn']

    by_name = {state.name: state for state in states}
    melt = {}
    floating = {}
    for state in states:
        filename = f'melt_{melt_form}_{state.name}.nc'
        melt[state.name], floating[state.name] = \
            _melt_from_run(filename, reference)

    # J1 and J2 come from the present-day climatology
    pd_melt = melt['climatology']
    pd_floating = floating['climatology']
    t1 = integrate_by_group(pd_melt, area, pd_floating, basins, CELL_DIMS,
                            group_dim='basins')
    t2 = integrate_by_group(pd_melt, area, pd_floating, bfrn, CELL_DIMS,
                            group_dim='BFRN_bins')

    # J3: the warm-minus-cold basin-mean difference, per ocean model
    means = {}
    for which in ('cold', 'warm'):
        per_model = []
        for state in states:
            if state.kind != 'model' or state.state != which:
                continue
            values = melt[state.name]
            # regional models only constrain the basins they cover
            if state.basins is not None:
                values = values.where(basins.isin(list(state.basins)))
            agg = average_by_group(values, area, floating[state.name],
                                   basins, CELL_DIMS, group_dim='basins')
            per_model.append(agg.expand_dims({'model': [state.label]}))
        means[which] = xr.concat(per_model, dim='model', coords='minimal')
    t3 = means['warm'] - means['cold']

    # J4: PIG and Dotson integrated melt, per observation year
    per_year = []
    for state in states:
        if state.kind != 'obs':
            continue
        agg = integrate_by_group(melt[state.name], area,
                                 floating[state.name],
                                 static['region_label'], CELL_DIMS,
                                 group_dim='region')
        per_year.append(agg.expand_dims({'year': [state.year]}))
    t4 = xr.concat(per_year, dim='year', coords='minimal')
    # drop the '' label used for cells outside PIG and Dotson
    t4 = t4.sel(region=[region for region in t4.region.values if region])

    logger.info(f'  total present-day melt at the reference parameter: '
                f'{float(t1.sum()) * reference:.1f} Gt/yr')

    ds = xr.Dataset({'t1': t1, 't2': t2, 't3': t3, 't4': t4})
    ds.attrs['melt_form'] = melt_form
    ds.attrs['reference_parameter'] = reference
    ds.attrs['note'] = (
        'Aggregates at a unit melt parameter.  Melt is exactly proportional '
        'to the parameter, so the whole parameter ensemble is these times '
        'each parameter value.')
    ds.attrs['ocean_states'] = ', '.join(sorted(by_name))
    return ds


def _report_shelf_area(static, logger):
    """Report MALI's per-basin shelf area against the observed extent."""
    area = static['area']
    basins = static['basins']
    observed = static['observed_floating']

    modelled_file = None
    for name in ('melt_ismip7_climatology.nc', 'melt_ismip6_climatology.nc'):
        try:
            modelled = melt_model.read_run(name)['floating']
            modelled_file = name
            break
        except FileNotFoundError:
            continue
    if modelled_file is None:
        logger.warning('No melt output found; skipping the shelf-area report')
        return

    grouper = basins.rename('basins')
    modelled_area = (area.where(modelled).groupby(grouper).sum() / 1.0e9)
    observed_area = (area.where(observed).groupby(grouper).sum() / 1.0e9)

    logger.info('')
    logger.info('Ice-shelf area by ISMIP7 basin, 10^3 km^2:')
    logger.info(f'  {"basin":>5s} {"MALI":>10s} {"observed":>10s} '
                f'{"ratio":>8s}')
    for basin in modelled_area.basins.values:
        mali = float(modelled_area.sel(basins=basin))
        obs = float(observed_area.sel(basins=basin))
        ratio = mali / obs if obs > 0 else float('nan')
        logger.info(f'  {int(basin):5d} {mali:10.1f} {obs:10.1f} '
                    f'{ratio:8.2f}')
    total_mali = float(modelled_area.sum())
    total_obs = float(observed_area.sum())
    logger.info(f'  {"total":>5s} {total_mali:10.1f} {total_obs:10.1f} '
                f'{total_mali / total_obs:8.2f}')
    logger.info('  (J1, J2 and J4 are integrals, so a shelf-area mismatch '
                'enters the calibrated parameter directly; J3 is a mean and '
                'is much less sensitive)')
    logger.info('')

    ds = xr.Dataset({'modelled_shelf_area': modelled_area,
                     'observed_shelf_area': observed_area})
    ds['modelled_shelf_area'].attrs['units'] = '10^3 km^2'
    ds['observed_shelf_area'].attrs['units'] = '10^3 km^2'
    write_netcdf(ds, 'shelf_area.nc')


def unit_aggregates(filename):
    """
    Read the unit aggregates written by this step.

    Parameters
    ----------
    filename : str
        An ``aggregates_<melt_form>.nc`` file

    Returns
    -------
    units : dict
        ``t1``, ``t2``, ``t3`` and ``t4``
    """
    ds = xr.load_dataset(filename)
    return {name: ds[name] for name in ('t1', 't2', 't3', 't4')}


def basin_coordinate(units):
    """
    The ISMIP7 basin numbers present in an aggregate, as integers.

    Parameters
    ----------
    units : dict
        From :py:func:`unit_aggregates`

    Returns
    -------
    basins : numpy.ndarray
        The basin numbers
    """
    return np.asarray(units['t1'].basins.values, dtype=int)
