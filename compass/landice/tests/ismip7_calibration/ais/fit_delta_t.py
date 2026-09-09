"""
Fit the per-basin thermal-forcing correction dT_b, after parameter selection.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration.ais import melt_model
from compass.landice.tests.ismip7_calibration.configure import parameter_name
from compass.step import Step


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

    For each basin, ``dT`` is searched on a grid within the configured
    bounds, minimising the absolute difference between the basin-integrated
    melt and the IMBIE observed value.  The protocol recommends bounding
    ``dT_b`` by about 2 K, since the warmest-to-coldest spread across
    Antarctic ice shelves is only 3-4 K, and recommends avoiding the
    correction where possible since it is "only an ad-hoc correction".

    The search is done on the MALI mesh with **area-weighted** integrals.
    The toolbox's own ``optimise_deltaT`` cannot be used, because it sums
    over structured-grid ``x`` and ``y`` dimensions with a single scalar cell
    area.

    Melt is recomputed in Python from MALI's ``TFdraft``, so no further MALI
    runs are needed: ``dT`` enters the melt expression only through the
    thermal forcing.

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
            self.add_output_file(filename=f'melt_params_{melt_form}.nc')

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
        observed = _load_observed(base_path)

        section = config['ismip7_calibration_delta_t']
        delta_t_grid = np.linspace(section.getfloat('delta_t_min'),
                                   section.getfloat('delta_t_max'),
                                   section.getint('delta_t_count'))

        ds_masks = xr.open_dataset('masks.nc')

        for melt_form in self.melt_forms:
            with xr.open_dataset(f'calibration_{melt_form}.nc') as ds_calib:
                parameter = float(ds_calib['median'])
            name = parameter_name(melt_form)
            logger.info(f'Fitting dT_b for the {melt_form} form at the '
                        f'median {name} = {parameter:.5e}')

            fields = melt_model.read_run(f'melt_{melt_form}.nc',
                                         'mesh.nc')
            result = _fit(melt_form, parameter, fields, ds_masks, observed,
                          delta_t_grid, config, logger)
            _write_params(result, ds_masks, melt_form, parameter, name,
                          f'melt_params_{melt_form}.nc')


def _load_observed(base_path):
    """The IMBIE basin-integrated melt targets, in Gt yr-1."""
    path = os.path.join(base_path, 'parameterisations', 'ocean', 'meltobs',
                        'Melt_Paolo_Davison_Adusumilli_imbie2.csv')
    return pd.read_csv(path, index_col=0)['BMR (Gt/yr)']


def _fit(melt_form, parameter, fields, ds_masks, observed, delta_t_grid,
         config, logger):
    """Grid-search dT per basin against the observed integrated melt."""
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

    basins = sorted(int(value) for value in np.unique(basin.values)
                    if value >= 0)

    optimal = {}
    residual = {}
    modelled_zero = {}
    for basin_number in basins:
        if basin_number not in observed.index:
            continue
        target = float(observed.loc[basin_number])
        in_basin = (basin == basin_number) & floating
        if not bool(in_basin.any()):
            continue

        totals = []
        for delta_t in delta_t_grid:
            melt = melt_model.melt_from_tf(melt_form, parameter, fields,
                                           config, delta_t=delta_t,
                                           mean_tf=mean_tf)
            totals.append(float((melt * area).where(in_basin).sum()) / 1.0e12)
        totals = np.asarray(totals)

        best = int(np.argmin(np.abs(totals - target)))
        optimal[basin_number] = float(delta_t_grid[best])
        residual[basin_number] = float(abs(totals[best] - target))
        zero = int(np.argmin(np.abs(delta_t_grid)))
        modelled_zero[basin_number] = float(totals[zero])

    logger.info('')
    logger.info(f'dT_b per ISMIP7 basin, {melt_form}:')
    logger.info(f'  {"basin":>5s} {"dT_b (K)":>9s} {"melt(0)":>10s} '
                f'{"observed":>10s} {"residual":>10s}')
    for basin_number in sorted(optimal):
        logger.info(f'  {basin_number:5d} {optimal[basin_number]:9.3f} '
                    f'{modelled_zero[basin_number]:10.1f} '
                    f'{float(observed.loc[basin_number]):10.1f} '
                    f'{residual[basin_number]:10.2f}')
    at_bound = [basin_number for basin_number, value in optimal.items()
                if abs(value) >= 0.999 * max(abs(delta_t_grid[0]),
                                             abs(delta_t_grid[-1]))]
    if at_bound:
        logger.warning(f'  dT_b hit the bounds in basins {at_bound}; the '
                       f'protocol bounds the correction deliberately, so a '
                       f'basin at the bound means the melt form cannot match '
                       f'the observed integral there.')
    logger.info('')
    return optimal


def _write_params(optimal, ds_masks, melt_form, parameter, name, filename):
    """Write a MALI melt-parameter file carrying the fitted dT_b."""
    basin0 = ds_masks['ismip7BasinNumber']
    delta_t = xr.zeros_like(basin0, dtype=float)
    for basin_number, value in optimal.items():
        delta_t = xr.where(basin0 == basin_number, value, delta_t)

    ds = xr.Dataset()
    ds['ismip6shelfMelt_basin'] = ds_masks['ismip6shelfMelt_basin']
    ds['ismip6shelfMelt_deltaT'] = delta_t
    ds['ismip6shelfMelt_deltaT'].attrs = {
        'long_name': 'basin-wide thermal forcing correction',
        'units': 'degC'}
    if melt_form == 'ismip6':
        ds['ismip6shelfMelt_gamma0'] = parameter
        ds['ismip6shelfMelt_gamma0'].attrs = {'units': 'm yr^-1'}

    ds.attrs['melt_form'] = melt_form
    ds.attrs['parameter_name'] = name
    ds.attrs['parameter_value'] = parameter
    ds.attrs['note'] = (
        'dT_b fitted after parameter selection, per protocol Sect. 4.2.1 '
        'option 2, against the IMBIE basin-integrated melt observations.')
    write_netcdf(ds, filename)
