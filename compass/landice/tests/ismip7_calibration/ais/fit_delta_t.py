"""
Fit the per-basin thermal-forcing correction dT_b, after parameter selection.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration.ais import melt_model
from compass.landice.tests.ismip7_calibration.ais.run_state import (
    PARAMETER_VARIABLE,
)
from compass.landice.tests.ismip7_calibration.configure import (
    is_ismip7,
    parameter_name,
)
from compass.step import Step

#: the percentiles of the parameter distribution a parameter file is
#: written for, and the suffix that names each file
PERCENTILES = (('p5', 'p05'), ('median', 'p50'), ('p95', 'p95'))


class FitDeltaT(Step):
    """
    A step that fits the basin thermal-forcing correction ``dT_b``.

    Protocol Sect. 4.2.1 permits the corrections to be calculated at either
    stage: **(1)** before the parameter optimisation, which "can lead to the
    problem that the parameter bounds are not constrained when present-day
    melt rates are compared to observations (namely, terms J1 and J2)"; or
    **(2)** after it, "as in ISMIP6".  This step follows **(2)**, which is
    what the published quadratic worked example does, and it is why the
    calibration itself runs with ``dT_b = 0`` throughout.

    ``dT_b`` depends on the melt parameter, so it is fitted separately at the
    5th, 50th and 95th percentiles of the selected distribution, and a
    complete MALI parameter file -- basins, corrections and the parameter
    itself -- is written for each.  Projections at each percentile then
    need only point at the matching file.

    For each basin, ``dT`` is searched on a grid within the configured
    bounds, minimising the absolute difference between the basin-integrated
    melt and the IMBIE observed value.  The protocol recommends bounding
    ``dT_b`` by about 2 K, since the warmest-to-coldest spread across
    Antarctic ice shelves is only 3-4 K, and recommends avoiding the
    correction where possible since it is "only an ad-hoc correction".

    The search is done on the MALI mesh with **area-weighted** integrals.
    The toolbox's own ``optimise_deltaT`` cannot be used, because it sums
    over structured-grid ``x`` and ``y`` dimensions with a single scalar cell
    area.  Melt is recomputed in Python from MALI's ``TFdraft``, so no
    further MALI runs are needed: ``dT`` enters the melt expression only
    through the thermal forcing.

    Attributes
    ----------
    melt_forms : list of str
        The melt forms to fit dT_b for
    """

    def __init__(self, test_case, melt_forms, state_name='climatology'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        melt_forms : list of str
            The melt forms to fit dT_b for

        state_name : str, optional
            The ocean state to fit against; the present-day climatology, to
            match the observed present-day melt
        """
        super().__init__(test_case=test_case, name='fit_delta_t')
        self.melt_forms = melt_forms
        self.state_name = state_name

        self.add_input_file(filename='masks.nc',
                            target='../remap_masks/ismip7_masks_on_mali.nc')
        for melt_form in melt_forms:
            self.add_input_file(
                filename=f'melt_{melt_form}.nc',
                target=f'../{melt_form}_{state_name}/output_melt.nc')
            self.add_input_file(
                filename=f'calibration_{melt_form}.nc',
                target=f'../calibrate/calibration_{melt_form}.nc')
            for _, suffix in PERCENTILES:
                self.add_output_file(
                    filename=f'melt_params_{melt_form}_{suffix}.nc')
            self.add_output_file(filename=f'delta_t_{melt_form}.nc')

    def setup(self):
        """
        Set up this step of the test case
        """
        section = self.config['ismip7_calibration']
        self.add_input_file(
            filename='mesh.nc',
            target=(f'{section.get("base_path_mali")}/'
                    f'{section.get("mali_mesh_file")}'))

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config

        if not config.getboolean('ismip7_calibration_delta_t',
                                 'fit_delta_t'):
            logger.info('fit_delta_t is False; not fitting dT_b')
            return

        base_path = config.get('ismip7_calibration', 'base_path_ismip7')
        observed = load_observed_melt(base_path)

        section = config['ismip7_calibration_delta_t']
        delta_t_grid = np.linspace(section.getfloat('delta_t_min'),
                                   section.getfloat('delta_t_max'),
                                   section.getint('delta_t_count'))

        ds_masks = xr.open_dataset('masks.nc')

        for melt_form in self.melt_forms:
            name = parameter_name(melt_form)
            fields = melt_model.read_run(f'melt_{melt_form}.nc', 'mesh.nc')
            results = {}
            with xr.open_dataset(f'calibration_{melt_form}.nc') as ds_calib:
                for percentile, suffix in PERCENTILES:
                    parameter = float(ds_calib[percentile])
                    logger.info(f'Fitting dT_b for the {melt_form} form at '
                                f'the {percentile} {name} = {parameter:.5e}')
                    fit = fit_basins(melt_form, parameter, fields, ds_masks,
                                     observed, delta_t_grid, config)
                    _report(fit, melt_form, delta_t_grid, logger)
                    _write_params(fit, ds_masks, melt_form, parameter, name,
                                  f'melt_params_{melt_form}_{suffix}.nc',
                                  config)
                    results[percentile] = (parameter, fit)
            _write_summary(results, melt_form, name,
                           f'delta_t_{melt_form}.nc', config)


def load_observed_melt(base_path):
    """
    The IMBIE basin-integrated melt targets, in Gt yr-1.

    Parameters
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets

    Returns
    -------
    observed : pandas.Series
        Observed melt per ISMIP7 (0-based) basin
    """
    path = os.path.join(base_path, 'parameterisations', 'ocean', 'meltobs',
                        'Melt_Paolo_Davison_Adusumilli_imbie2.csv')
    return pd.read_csv(path, index_col=0)['BMR (Gt/yr)']


def fit_basins(melt_form, parameter, fields, ds_masks, observed, delta_t_grid,
               config):
    """
    Grid-search dT_b in every basin against the observed integrated melt.

    Parameters
    ----------
    melt_form : {'ismip7', 'ismip6'}
        The melt form

    parameter : float
        The value of the melt parameter to fit at

    fields : dict
        From :py:func:`compass.landice.tests.ismip7_calibration.ais.melt_model.read_run`

    ds_masks : xarray.Dataset
        The remapped ISMIP7 masks

    observed : pandas.Series
        From :py:func:`load_observed_melt`

    delta_t_grid : numpy.ndarray
        The dT values to search, in K

    config : compass.config.CompassConfigParser
        Configuration options, for the melt-form constants

    Returns
    -------
    fit : dict
        Per ISMIP7 basin number: ``delta_t``, ``observed``,
        ``modelled_before`` (at dT = 0), ``modelled_after`` (at the fitted
        dT) and ``residual``, all melt in Gt yr-1
    """  # noqa: E501
    area = fields['area']
    floating = fields['floating']
    # the ISMIP7 0-based numbering, which is what the observations use
    basin = ds_masks['ismip7BasinNumber']

    # the basin mean of the ISMIP6 form shifts with dT exactly as the local
    # forcing does, because dT is constant within a basin
    mean_tf = None
    if melt_form == 'ismip6':
        mean_tf = melt_model.basin_mean_tf(fields['tf_draft'], area,
                                           floating, fields['basin'])

    # melt at every dT on the grid, once, then integrate per basin
    totals = {}
    for delta_t in delta_t_grid:
        melt = melt_model.melt_from_tf(melt_form, parameter, fields, config,
                                       delta_t=delta_t, mean_tf=mean_tf)
        totals[delta_t] = melt_model.integrate_by_basin(
            melt, area, floating, basin.where(basin >= 0))

    fit = {}
    for basin_number in sorted(int(b) for b in np.unique(basin.values)
                               if b >= 0):
        if basin_number not in observed.index:
            continue
        series = np.array([float(totals[dt].sel(basin=basin_number))
                           if basin_number in totals[dt].basin.values
                           else np.nan for dt in delta_t_grid])
        if not np.isfinite(series).any():
            continue
        fit[basin_number] = fit_one_basin(
            series, float(observed.loc[basin_number]), delta_t_grid)
    return fit


def fit_one_basin(modelled, observed, delta_t_grid):
    """
    Pick the dT that brings one basin's integrated melt closest to observed.

    Parameters
    ----------
    modelled : numpy.ndarray
        Basin-integrated melt at each dT in ``delta_t_grid``, Gt yr-1

    observed : float
        The observed basin-integrated melt, Gt yr-1

    delta_t_grid : numpy.ndarray
        The dT values, in K

    Returns
    -------
    fit : dict
        ``delta_t``, ``observed``, ``modelled_before``, ``modelled_after``
        and ``residual``
    """
    misfit = np.abs(modelled - observed)
    best = int(np.nanargmin(misfit))
    zero = int(np.argmin(np.abs(delta_t_grid)))
    return dict(delta_t=float(delta_t_grid[best]),
                observed=float(observed),
                modelled_before=float(modelled[zero]),
                modelled_after=float(modelled[best]),
                residual=float(misfit[best]))


def check_fit(fit, delta_t_grid):
    """
    Check that a fit did what it claims.

    Fitting dT_b must never make a basin worse than dT = 0, and the fitted
    values must lie within the grid.  Neither can fail by construction, so
    a failure here means a sign or indexing error upstream -- which is the
    kind of bug that otherwise produces plausible-looking corrections.

    Parameters
    ----------
    fit : dict
        From :py:func:`fit_basins`

    delta_t_grid : numpy.ndarray
        The dT values that were searched

    Raises
    ------
    ValueError
        If any basin got worse, or any fitted dT is off the grid
    """
    low, high = float(delta_t_grid.min()), float(delta_t_grid.max())
    problems = []
    for basin_number, result in fit.items():
        before = abs(result['modelled_before'] - result['observed'])
        if result['residual'] > before * (1.0 + 1.0e-12):
            problems.append(f'basin {basin_number}: residual '
                            f'{result["residual"]:.3g} is worse than '
                            f'{before:.3g} at dT = 0')
        if not low <= result['delta_t'] <= high:
            problems.append(f'basin {basin_number}: dT = '
                            f'{result["delta_t"]} is outside '
                            f'[{low}, {high}]')
    if problems:
        raise ValueError('The dT_b fit is inconsistent:\n  ' +
                         '\n  '.join(problems))


def _report(fit, melt_form, delta_t_grid, logger):
    """Log the fit per basin, and check it."""
    check_fit(fit, delta_t_grid)

    logger.info(f'  {"basin":>5s} {"dT_b (K)":>9s} {"melt(0)":>10s} '
                f'{"melt(dT)":>10s} {"observed":>10s} {"residual":>10s}')
    for basin_number in sorted(fit):
        result = fit[basin_number]
        logger.info(f'  {basin_number:5d} {result["delta_t"]:9.3f} '
                    f'{result["modelled_before"]:10.1f} '
                    f'{result["modelled_after"]:10.1f} '
                    f'{result["observed"]:10.1f} '
                    f'{result["residual"]:10.2f}')

    bound = max(abs(float(delta_t_grid.min())),
                abs(float(delta_t_grid.max())))
    at_bound = [basin_number for basin_number, result in fit.items()
                if abs(result['delta_t']) >= 0.999 * bound]
    if at_bound:
        logger.warning(f'  dT_b hit the bounds in basins {at_bound}; the '
                       f'protocol bounds the correction deliberately, so a '
                       f'basin at the bound means the melt form cannot match '
                       f'the observed integral there.')
    logger.info('')


def _write_params(fit, ds_masks, melt_form, parameter, name, filename,
                  config):
    """
    Write a complete MALI melt-parameter file for one percentile.

    It carries the basins, the fitted corrections and the melt parameter
    itself, so that a projection needs only to point at it.
    """
    basin0 = ds_masks['ismip7BasinNumber']
    delta_t = xr.zeros_like(basin0, dtype=float)
    for basin_number, result in fit.items():
        delta_t = xr.where(basin0 == basin_number, result['delta_t'],
                           delta_t)

    ds = xr.Dataset()
    ds['ismip6shelfMelt_basin'] = ds_masks['ismip6shelfMelt_basin']
    ds['ismip6shelfMelt_deltaT'] = delta_t
    ds['ismip6shelfMelt_deltaT'].attrs = {
        'long_name': 'basin-wide thermal forcing correction',
        'units': 'degC'}
    ds[PARAMETER_VARIABLE[melt_form]] = parameter
    ds[PARAMETER_VARIABLE[melt_form]].attrs = {
        'long_name': f'calibrated {name} for the {melt_form} ice-shelf '
                     f'melting method'}
    if melt_form == 'ismip6':
        ds[PARAMETER_VARIABLE[melt_form]].attrs['units'] = 'm yr^-1'

    ds.attrs['melt_form'] = melt_form
    ds.attrs['parameter_name'] = name
    ds.attrs['parameter_value'] = parameter
    ds.attrs['note'] = (
        'dT_b fitted after parameter selection, per protocol Sect. 4.2.1 '
        'option 2, against the IMBIE basin-integrated melt observations.')
    # Add slope configuration metadata for ISMIP7 forms
    if is_ismip7(melt_form):
        ds.attrs.update(melt_model.slope_metadata(config, melt_form))
    write_netcdf(ds, filename)


def _write_summary(results, melt_form, name, filename, config):
    """Write the fitted dT_b per basin and percentile, for validation."""
    percentiles = [percentile for percentile, _ in PERCENTILES]
    basins = sorted({basin_number for _, fit in results.values()
                     for basin_number in fit})

    def table(key):
        return np.array([[results[p][1][b][key] for b in basins]
                         for p in percentiles])

    ds = xr.Dataset(coords={'percentile': percentiles, 'basin': basins})
    ds['parameter'] = ('percentile',
                       [results[p][0] for p in percentiles])
    ds['delta_t'] = (('percentile', 'basin'), table('delta_t'))
    ds['modelled_before'] = (('percentile', 'basin'),
                             table('modelled_before'))
    ds['modelled_after'] = (('percentile', 'basin'), table('modelled_after'))
    ds['observed'] = ('basin', [results[percentiles[0]][1][b]['observed']
                                for b in basins])
    ds['delta_t'].attrs['units'] = 'degC'
    for key in ('modelled_before', 'modelled_after', 'observed'):
        ds[key].attrs['units'] = 'Gt yr^-1'
    ds.attrs['melt_form'] = melt_form
    ds.attrs['parameter_name'] = name
    # Add slope configuration metadata for ISMIP7 forms
    if is_ismip7(melt_form):
        ds.attrs.update(melt_model.slope_metadata(config, melt_form))
    write_netcdf(ds, filename)
