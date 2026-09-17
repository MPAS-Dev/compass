"""
Reproduce the published 8 km ISMIP7 quadratic calibration.
"""

import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration import datasets
from compass.landice.tests.ismip7_calibration.configure import (
    objective_options,
)
from compass.landice.tests.ismip7_calibration.objective import (
    build_toolbox_terms,
    run_optimisation,
)
from compass.landice.tests.ismip7_calibration.quadratic import (
    local_quadratic_melt,
    mean_slope,
)
from compass.landice.tests.ismip7_calibration.terms import (
    average_by_group,
    integrate_by_group,
    uniform_area,
)
from compass.step import Step

#: the ISMIP grid resolution the published calibration used, m
RESOLUTION = 8000.0

#: cell dimensions of the ISMIP structured grid
CELL_DIMS = ('y', 'x')

#: the 120 K values the protocol's worked example samples
K_VALUES = np.arange(0.25e-5, 3.025e-4, 0.25e-5)

#: percentiles published in protocol Sect. 4.3.1 and Fig. 5
PUBLISHED = {'p5': 4.75e-5, 'median': 8.5e-5, 'p95': 1.375e-4}


class Replicate(Step):
    """
    A step that reproduces the published 8 km quadratic calibration.

    Attributes
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.replication.Replication
            The test case this step belongs to
        """  # noqa: E501
        super().__init__(test_case=test_case, name='replicate')
        self.base_path = None
        self.add_output_file(filename='replication_8km.nc')

    def setup(self):
        """
        Check that the ISMIP7 datasets this step needs are present
        """
        config = self.config
        self.base_path = config.get('ismip7_calibration', 'base_path_ismip7')

        states = datasets.ocean_states(self.base_path, subset='all')
        missing = datasets.missing_files(states)
        if missing:
            listing = '\n  '.join(f'{name}: {path}' for name, path in missing)
            raise FileNotFoundError(
                f'{len(missing)} ISMIP7 input files are missing:\n  '
                f'{listing}')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        base_path = config.get('ismip7_calibration', 'base_path_ismip7')

        static = _load_static(base_path, logger)
        units = _build_unit_terms(base_path, static, logger)
        targets = datasets.load_targets(base_path)

        # The published example weights four ocean models, and PIG alone in
        # 2009 and 2012.  Reproducing the published numbers means reproducing
        # that weighting, so it is fixed here rather than taken from config.
        terms = build_toolbox_terms(
            units, targets, K_VALUES,
            t3_models=datasets.PUBLISHED_T3_MODELS,
            t4_regions=datasets.PUBLISHED_T4_REGIONS,
            t4_years=datasets.PUBLISHED_T4_YEARS)

        sample_size, seed = objective_options(config)
        logger.info(f'Sampling the objective function {sample_size} times...')
        result = run_optimisation(terms, K_VALUES, resolution=RESOLUTION,
                                  sample_size=sample_size, seed=seed)

        _write_result(result, K_VALUES, 'replication_8km.nc')
        _report(result, logger)


def check_published(filename, logger):
    """
    Check the replicated percentiles against the published values.

    Parameters
    ----------
    filename : str
        The file written by :py:class:`Replicate`

    logger : logging.Logger
        A logger to report the comparison to

    Raises
    ------
    ValueError
        If any percentile differs from the published value
    """
    result = xr.open_dataset(filename)
    mismatched = []
    for name, expected in PUBLISHED.items():
        found = float(result[name])
        # the published values are points on the parameter grid, so this
        # should be exact to within floating-point representation rather
        # than merely close
        if not np.isclose(found, expected, rtol=1.0e-9, atol=0.0):
            mismatched.append(f'  {name}: expected {expected:.5e}, '
                              f'found {found:.5e}')
    result.close()

    if mismatched:
        listing = '\n'.join(mismatched)
        raise ValueError(
            f'The replication of the published 8 km ISMIP7 calibration does '
            f'not reproduce the published percentiles:\n{listing}\n'
            f'This means the vendored parameter-selection toolbox is not '
            f'being driven as it was for the published numbers, so the MALI '
            f'calibration built on the same code path cannot be trusted '
            f'either.')

    logger.info('The published 8 km percentiles are reproduced exactly.')


def _load_static(base_path, logger):
    """Load the 8 km topography, masks and derived fields."""
    logger.info('Loading the ISMIP 8 km topography and masks...')
    masks = datasets.mask_files(base_path)

    topo = xr.load_dataset(
        os.path.join(base_path, 'obs', 'ocean', 'topography', 'bedmap3', 'v3',
                     'bedmap3_AIS_obs_ocean_topography_v3.nc'))
    basins = xr.load_dataset(masks['basins']).basinNumber.rename('basins')
    bfrn = xr.load_dataset(masks['bfrn'])
    mask = xr.load_dataset(masks['floating']).mask

    floating = topo['floating_frac'] > 0.5
    slope = mean_slope(topo['draft'], floating, RESOLUTION, RESOLUTION)
    logger.info(f'  mean draft slope: {slope:.6f} rad '
                f'(sin = {np.sin(slope):.7f})')

    shelves = xr.load_dataset(masks['shelves']).shelf_mask.isel(time=0)
    x = shelves['x'] if 'x' in shelves.coords else basins['x']
    # restrict Pine Island to its main trunk, as the worked example does
    pig = (shelves == datasets.PIG_ID) & (x > datasets.PIG_X_MAX)
    dotson = shelves == datasets.DOTSON_ID
    region_label = xr.where(pig, 'pig', xr.where(dotson, 'dotson', ''))

    return dict(draft=topo['draft'], floating=floating, basins=basins,
                bfrn=bfrn, mask=mask, slope=slope, region_label=region_label,
                area=uniform_area(topo['draft'], RESOLUTION, CELL_DIMS))


def _unit_melt(state, static):
    """
    Melt at a unit ``K`` for one ocean state, in kg m-2 yr-1.

    Melt is exactly linear in ``K``, so every member of the parameter
    ensemble is this field times its ``K``.  Working at ``K = 1`` keeps the
    whole replication in single fields rather than 120 copies of each.
    """
    draft = static['draft']
    floating = static['floating']

    tf = xr.load_dataset(state.tf_file)['tf']
    so = xr.load_dataset(state.so_file)['so']

    tf_draft = tf.sel(z=draft, method='nearest').where(floating)
    so_draft = so.sel(z=draft, method='nearest').where(floating)

    melt = local_quadratic_melt(1.0, tf_draft, so_draft, static['slope'])
    # the observational files carry a scalar Time coordinate that differs
    # between years, which would collide when the per-year aggregates are
    # concatenated; nothing downstream uses any non-dimension coordinate
    return melt.drop_vars([name for name in melt.coords
                           if name not in melt.dims], errors='ignore')


def _build_unit_terms(base_path, static, logger):
    """Aggregate unit melt into the four objective-function terms."""
    area, mask = static['area'], static['mask']
    basins, bfrn = static['basins'], static['bfrn']
    states = datasets.ocean_states(base_path, subset='all')
    by_name = {state.name: state for state in states}

    logger.info('Building the present-day ensemble (J1, J2)...')
    pd_unit = _unit_melt(by_name['climatology'], static)
    t1 = integrate_by_group(pd_unit, area, mask, basins, CELL_DIMS,
                            group_dim='basins')
    t2 = integrate_by_group(pd_unit, area, mask, bfrn['BFRN_bins'],
                            CELL_DIMS, group_dim='BFRN_bins')

    logger.info('Building the ocean-model ensemble (J3)...')
    means = {}
    for which in ('cold', 'warm'):
        per_model = []
        for state in states:
            if state.kind != 'model' or state.state != which:
                continue
            melt = _unit_melt(state, static)
            # regional models only constrain the basins they cover
            if state.basins is not None:
                melt = melt.where(basins.isin(list(state.basins)))
            agg = average_by_group(melt, area, mask, basins, CELL_DIMS,
                                   group_dim='basins')
            per_model.append(agg.expand_dims({'model': [state.label]}))
            logger.info(f'  {which:4s} {state.label}')
        means[which] = xr.concat(per_model, dim='model',
                                 coords='minimal')
    t3 = means['warm'] - means['cold']

    logger.info('Building the observational ensemble (J4)...')
    per_year = []
    for state in states:
        if state.kind != 'obs':
            continue
        melt = _unit_melt(state, static)
        agg = integrate_by_group(melt, area, mask, static['region_label'],
                                 CELL_DIMS, group_dim='region')
        per_year.append(agg.expand_dims({'year': [state.year]}))
        logger.info(f'  {state.year}')
    t4 = xr.concat(per_year, dim='year', coords='minimal')
    # drop the '' label used for cells outside PIG and Dotson
    t4 = t4.sel(region=[region for region in t4.region.values if region])

    return dict(t1=t1, t2=t2, t3=t3, t4=t4)


def _write_result(result, param_values, filename):
    """Write the parameter distribution to a file."""
    ds = xr.Dataset()
    ds['min_p1'] = ('sample', result['min_p1'])
    ds['parameter_values'] = ('parameter', np.asarray(param_values))
    for name in ('p5', 'median', 'p95', 'mode'):
        ds[name] = float(result[name])
    ds.attrs['description'] = (
        'Replication of the published ISMIP7 quadratic calibration on the '
        'ISMIP 8 km grid')
    write_netcdf(ds, filename)


def _report(result, logger):
    """Log the percentiles beside the published values."""
    logger.info('')
    logger.info('K percentiles, ISMIP 8 km grid:')
    logger.info(f'{"":12s}{"compass":>12s}{"published":>12s}')
    for name, expected in PUBLISHED.items():
        logger.info(f'{name:12s}{result[name]:12.5e}{expected:12.5e}')
    logger.info(f'{"mode":12s}{result["mode"]:12.5e}')
    logger.info('')
