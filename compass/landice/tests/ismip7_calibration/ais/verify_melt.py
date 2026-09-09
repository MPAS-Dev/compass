"""
Verify MALI's melt against the Python reference implementation.
"""

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration.ais import melt_model
from compass.step import Step

#: the melt expression should agree to round-off
MELT_TOLERANCE = 1.0e-10

#: so should the vertical interpolation of thermal forcing, in K
INTERPOLATION_TOLERANCE = 1.0e-10

#: melt should be exactly proportional to the melt parameter; allow only
#: floating-point noise
LINEARITY_TOLERANCE = 1.0e-12


class VerifyMelt(Step):
    """
    A step that checks MALI's melt field against an independent Python
    implementation of the same equations.

    Three things are checked, and they are deliberately independent:

    * the **melt expression**, by evaluating the Python reference on MALI's
      *own* ``TFdraft``.  That isolates the formula from the vertical
      interpolation that produced ``TFdraft``.
    * the **vertical interpolation**, by interpolating the 3-D forcing to the
      ice draft with a plain ``numpy`` implementation written from the
      protocol rather than transliterated from the Fortran, and comparing
      against MALI's ``TFdraft``.
    * the **linearity in the melt parameter**, which is what licenses one
      MALI run per ocean state instead of one per (state, parameter) pair.

    The draft used for the interpolation check is reconstructed from the
    mesh file, never taken from the run output: melt thins the ice over the
    single timestep, so the output geometry is post-step while ``TFdraft``
    was computed pre-step.

    Attributes
    ----------
    melt_form : str
        The melt form being verified

    state_name : str
        The ocean state used for the check
    """

    def __init__(self, test_case, melt_form, state_name):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        melt_form : {'ismip7', 'ismip6'}
            The melt form to verify

        state_name : str
            The ocean state to verify against
        """
        super().__init__(test_case=test_case, name='verify_melt')
        self.melt_form = melt_form
        self.state_name = state_name

        self.add_input_file(
            filename='output_melt.nc',
            target=f'../{melt_form}_{state_name}/output_melt.nc')
        self.add_input_file(
            filename='forcing.nc',
            target=f'../remap_forcing/forcing_{state_name}.nc')
        self.add_output_file(filename='verification.nc')

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
        section = config['ismip7_calibration_melt']
        reference = {'ismip7': section.getfloat('reference_k'),
                     'ismip6': section.getfloat('reference_gamma0')}

        fields = melt_model.read_run('output_melt.nc')
        results = {}

        results['melt'] = _check_melt_expression(
            self.melt_form, reference[self.melt_form], fields, config,
            logger)
        results['interpolation'] = _check_interpolation(fields, logger)
        results['linearity'] = _check_linearity(
            self.melt_form, reference[self.melt_form], fields, config,
            logger)

        ds = xr.Dataset({name: float(value)
                         for name, value in results.items()})
        ds.attrs['melt_form'] = self.melt_form
        ds.attrs['ocean_state'] = self.state_name
        write_netcdf(ds, 'verification.nc')

        _raise_on_failure(results, logger)


def _check_melt_expression(melt_form, parameter, fields, config, logger):
    """Compare MALI's melt with the Python reference on MALI's TFdraft."""
    expected = melt_model.melt_from_tf(melt_form, parameter, fields, config)
    actual = fields['melt']

    melting = fields['floating'] & (np.abs(actual) > 0.0)
    count = int(melting.sum())
    if count == 0:
        raise ValueError('MALI produced no melt anywhere, so the melt '
                         'expression cannot be checked.  Check that the run '
                         'took a real timestep and that the forcing was '
                         'read.')

    difference = np.abs(actual - expected).where(melting)
    scale = np.abs(actual).where(melting)
    relative = float((difference / scale).max())

    logger.info('')
    logger.info(f'Melt expression, {melt_form}:')
    logger.info(f'  melting cells                 {count}')
    logger.info(f'  MALI melt range               '
                f'{float(actual.where(melting).min()):.4g} .. '
                f'{float(actual.where(melting).max()):.4g} kg/m2/yr')
    logger.info(f'  max relative difference       {relative:.3e}')
    return relative


def _check_interpolation(fields, logger):
    """Compare MALI's TFdraft with an independent interpolation."""
    with xr.open_dataset('forcing.nc') as ds:
        tf_3d = ds['ismip6shelfMelt_3dThermalForcing'].isel(Time=0).values
        z_ocean = ds['ismip6shelfMelt_zOcean'].values

    draft = melt_model.initial_draft('mesh.nc').values
    with xr.open_dataset('mesh.nc') as ds_mesh:
        bed = ds_mesh['bedTopography']
        if 'Time' in bed.dims:
            bed = bed.isel(Time=0)
        bed = bed.values

    floating = fields['floating'].values
    expected = melt_model.interpolate_to_draft(tf_3d, z_ocean, draft, bed)
    actual = fields['tf_draft'].values

    # MALI applies a freezing-point depth correction below the deepest layer
    # centre that the reference here does not, so those cells are excluded
    deepest = z_ocean[-1]
    interior = floating & (draft > deepest)
    difference = np.abs(actual[interior] - expected[interior])
    largest = float(np.nanmax(difference)) if difference.size else 0.0

    logger.info('')
    logger.info('Vertical interpolation of thermal forcing:')
    logger.info(f'  cells compared                {int(interior.sum())}')
    logger.info(f'  max |difference|              {largest:.3e} K')
    return largest


def _check_linearity(melt_form, parameter, fields, config, logger):
    """
    Check that melt is exactly proportional to the melt parameter.

    This is measured rather than assumed, because it is what licenses one
    MALI run per ocean state instead of one per (state, parameter) pair --
    28 runs rather than about 1300.
    """
    factors = (0.5, 1.0, 2.0)
    totals = []
    for factor in factors:
        melt = melt_model.melt_from_tf(melt_form, parameter * factor,
                                       fields, config)
        total = float((melt * fields['area']).where(
            fields['floating']).sum()) / 1.0e12
        totals.append(total)

    per_unit = [total / factor for total, factor in zip(totals, factors)]
    spread = (max(per_unit) - min(per_unit)) / abs(per_unit[1])

    logger.info('')
    logger.info(f'Linearity in the melt parameter, {melt_form}:')
    for factor, total in zip(factors, totals):
        logger.info(f'  {factor:4.1f} x reference          '
                    f'{total:12.4f} Gt/yr')
    logger.info(f'  max relative deviation        {spread:.3e}')
    logger.info('')
    return spread


def _raise_on_failure(results, logger):
    """Raise if any check exceeded its tolerance."""
    checks = (('melt', MELT_TOLERANCE,
               "MALI's melt does not match the Python reference evaluated "
               "on MALI's own TFdraft"),
              ('interpolation', INTERPOLATION_TOLERANCE,
               "MALI's vertical interpolation of thermal forcing to the ice "
               "draft does not match an independent implementation"),
              ('linearity', LINEARITY_TOLERANCE,
               'melt is not exactly proportional to the melt parameter, so '
               'the ensemble cannot be built by scaling a single run per '
               'ocean state'))

    failures = [f'  {name}: {results[name]:.3e} exceeds {tol:.0e} -- {why}'
                for name, tol, why in checks if results[name] > tol]
    if failures:
        listing = '\n'.join(failures)
        raise ValueError(f'MALI melt verification failed:\n{listing}')

    logger.info('All melt verification checks passed.')
