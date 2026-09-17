"""
Registry of the ISMIP7 datasets used by the melt-module calibration.

One source of truth for *which* ocean states feed *which* objective-function
term, and for where the calibration targets live, shared by the 8 km
``replication`` test case and the MALI-mesh ``ais`` test case so the two
cannot drift apart.

The ocean-state sets follow protocol Table 2 and the focus group's 31 July
2026 update:

* ``minimal`` -- the states marked X in Table 2: present-day climatology, two
  circum-Antarctic cold/warm pairs, and PIG in 2009 and 2012.  7 states.
* ``recommended`` -- adds the two regional cold/warm pairs marked +, which the
  focus group strongly suggests.  11 states.
* ``all`` -- every state distributed for ISMIP7: all seven ocean models of
  protocol Table 2 (AIS1-AIS7) and all thirteen observation years.  28 states.
  This is what the ``model`` coordinate of the J3 target file expects, so only
  this subset exercises every target the protocol ships.
"""

import os
from collections import namedtuple

import numpy as np
import pandas as pd
import xarray as xr

#: ISMIP grid resolution of the calibration datasets, in km
ISMIP_RESOLUTION_KM = 8

#: ice-shelf ids in ``shelf_mask_ismip8km.nc``
PIG_ID = 110
DOTSON_ID = 97

#: Pine Island is cut at this x to keep only its main trunk, following the
#: protocol's worked example
PIG_X_MAX = -1.625e6

#: ocean-model datasets: file prefix, toolbox label, basins constrained.
#: ``basins`` is None where the simulation is circum-Antarctic; otherwise it
#: lists the ISMIP7 (0-based) basins the regional domain actually covers, and
#: melt outside them must be discarded.
OCEAN_MODELS = [
    ('Mathiot_NEMO_{state}_v3_', 'mathiot', None),
    ('Timmermann_FESOM_{state}_v3_', 'timmermann', (14,)),
    ('Naughten_FESOM_ACCESS_{state}_v2_', 'naughten_ais_1', None),
    ('Naughten_FESOM_MMM_{state}_v2_', 'naughten_ais_2', None),
    ('Jourdain-Naughten_NEMO-MITgcm_{state}_', 'jourdain_naughten', (9, 14)),
    ('Naughten_MITamu-MITwed_{state}_', 'naughten_naughten', (9, 14)),
    ('Haid_FESOM_{state}_', 'haid', (14,)),
]

#: all years of Amundsen near-ice-shelf ocean observations
OBS_YEARS = [1994, 2000, 2006, 2007, 2009, 2010, 2011, 2012, 2014, 2016,
             2018, 2019, 2020]

#: minimal J3 set (Table 2, marked X)
MINIMAL_MODELS = ('mathiot', 'naughten_ais_1')

#: J3 set the focus group recommends (adds those marked +)
RECOMMENDED_MODELS = ('mathiot', 'naughten_ais_1', 'jourdain_naughten',
                      'naughten_naughten')

#: J4 years the protocol suggests as a minimum, a cold and a warm PIG state
RECOMMENDED_OBS_YEARS = (2009, 2012)

#: the weighting the published quadratic example used: four ocean models, and
#: PIG alone in 2009 and 2012.  Narrower than protocol Table 2 makes available
PUBLISHED_T3_MODELS = RECOMMENDED_MODELS
PUBLISHED_T4_REGIONS = ('pig',)
PUBLISHED_T4_YEARS = RECOMMENDED_OBS_YEARS

#: One ocean state to force the melt module with.
#:
#: ``name`` is a unique, filesystem-safe identifier such as ``mathiot_cold``;
#: ``kind`` is ``'climatology'``, ``'model'`` or ``'obs'``; ``term`` names the
#: objective-function term the state feeds; ``label`` and ``state`` apply to
#: ocean-model states, ``year`` to observational ones; and ``basins`` lists
#: the ISMIP7 (0-based) basins a regional state constrains, or None for a
#: circum-Antarctic one.
OceanState = namedtuple(
    'OceanState',
    ['name', 'kind', 'tf_file', 'so_file', 'term', 'label', 'state', 'year',
     'basins'])


def climatology_files(base_path):
    """
    Paths to the Zhou et al. present-day climatology TF and salinity.

    Parameters
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets

    Returns
    -------
    tf_file : str
        The thermal-forcing file

    so_file : str
        The salinity file
    """
    clim = os.path.join(base_path, 'obs', 'ocean', 'climatology',
                        'zhou_annual_06_nov')
    stem = 'AIS_obs_ocean_climatology_zhou_annual_06_nov'
    return (os.path.join(clim, 'tf', 'v3', f'tf_{stem}_v3_1972-2024.nc'),
            os.path.join(clim, 'so', 'v4', f'so_{stem}_v4_1972-2024.nc'))


def ocean_states(base_path, subset='all'):
    """
    The ocean states to run the melt module for.

    Parameters
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets

    subset : {'minimal', 'recommended', 'all'}, optional
        Which set of states to include; see the module docstring

    Returns
    -------
    states : list of OceanState
        The ocean states in the requested subset
    """
    if subset not in ('minimal', 'recommended', 'all'):
        raise ValueError(f"subset must be 'minimal', 'recommended' or 'all', "
                         f"but is '{subset}'")

    param = os.path.join(base_path, 'parameterisations', 'ocean')
    model_dir = os.path.join(param, 'ocean_modelling_data')
    obs_dir = os.path.join(param, 'ocean_observations_data')

    if subset == 'minimal':
        keep_models = MINIMAL_MODELS
        keep_years = RECOMMENDED_OBS_YEARS
    elif subset == 'recommended':
        keep_models = RECOMMENDED_MODELS
        keep_years = RECOMMENDED_OBS_YEARS
    else:
        keep_models = tuple(label for _, label, _ in OCEAN_MODELS)
        keep_years = tuple(OBS_YEARS)

    tf_file, so_file = climatology_files(base_path)
    states = [OceanState(name='climatology', kind='climatology',
                         tf_file=tf_file, so_file=so_file, term='J1,J2',
                         label=None, state=None, year=None, basins=None)]

    for prefix, label, basins in OCEAN_MODELS:
        if label not in keep_models:
            continue
        for state in ('cold', 'warm'):
            stem = prefix.format(state=state)
            states.append(OceanState(
                name=f'{label}_{state}', kind='model',
                tf_file=os.path.join(model_dir, f'{stem}TF.nc'),
                so_file=os.path.join(model_dir, f'{stem}S.nc'),
                term='J3', label=label, state=state, year=None,
                basins=basins))

    for year in OBS_YEARS:
        if year not in keep_years:
            continue
        states.append(OceanState(
            name=f'obs_{year}', kind='obs',
            tf_file=os.path.join(obs_dir, f'Obs_{year}_TF.nc'),
            so_file=os.path.join(obs_dir, f'Obs_{year}_S.nc'),
            term='J4', label=None, state=None, year=year, basins=(9,)))

    return states


def missing_files(states):
    """
    The input files of ``states`` that are not present on disk.

    Parameters
    ----------
    states : list of OceanState
        The ocean states to check

    Returns
    -------
    missing : list of tuple
        ``(state name, path)`` pairs for each file that does not exist
    """
    missing = []
    for state in states:
        for path in (state.tf_file, state.so_file):
            if not os.path.exists(path):
                missing.append((state.name, path))
    return missing


def mask_files(base_path, resolution_km=ISMIP_RESOLUTION_KM):
    """
    Paths to the ISMIP7 mask datasets on the ISMIP polar stereographic grid.

    Parameters
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets

    resolution_km : int, optional
        ISMIP grid resolution in km

    Returns
    -------
    files : dict
        Paths keyed by ``basins``, ``bfrn``, ``floating`` and ``shelves``
    """
    param = os.path.join(base_path, 'parameterisations', 'ocean')
    res = resolution_km
    return dict(
        basins=os.path.join(param, 'imbie2',
                            f'basin_numbers_ismip{res}km_v2.nc'),
        bfrn=os.path.join(param, 'bfrns', f'BFRN_ismip{res}km_v2.nc'),
        floating=os.path.join(param, 'floatingmasks',
                              f'floatingmask_ismip{res}km.nc'),
        shelves=os.path.join(param, 'shelfmask',
                             f'shelf_mask_ismip{res}km.nc'))


def load_targets(base_path):
    """
    Load the observational targets for the four objective-function terms.

    These are on the ISMIP grid or are already aggregated, so they are the
    same whether the modelled melt came from the 8 km reference implementation
    or from MALI.

    Parameters
    ----------
    base_path : str
        Root of the ISMIP7 AIS datasets

    Returns
    -------
    targets : dict
        ``t1_mean``, ``t1_sigma``, ``t2_mean``, ``t2_sigma``, ``t2_weights``,
        ``t3_mean``, ``t3_sigma``, ``t4_mean`` and ``t4_sigma``
    """
    param = os.path.join(base_path, 'parameterisations', 'ocean')

    melt_imbie = pd.read_csv(
        os.path.join(param, 'meltobs',
                     'Melt_Paolo_Davison_Adusumilli_imbie2.csv'),
        index_col=0)
    basin_coord = np.arange(len(melt_imbie))
    t1_mean = xr.DataArray(
        melt_imbie['BMR (Gt/yr)'].values.astype(float),
        dims=['basin'], coords={'basin': basin_coord})
    t1_sigma = xr.DataArray(
        melt_imbie['BMR uncert (Gt/yr)'].values.astype(float),
        dims=['basin'], coords={'basin': basin_coord})

    buttressing = xr.load_dataset(
        os.path.join(param, 'meltobs', 'melt_target_term2_v3.nc'))
    bfrn = xr.load_dataset(mask_files(base_path)['bfrn'])
    t2_weights = xr.DataArray(
        (bfrn['BFRN_medians'] / bfrn['BFRN_median']).values,
        dims=['BFRN_bins'],
        coords={'BFRN_bins': buttressing.BFRN_bins.values})

    cold = xr.load_dataset(
        os.path.join(param, 'ocean_modelling_data',
                     'melt_cold_target_term3_v2.nc'))
    warm = xr.load_dataset(
        os.path.join(param, 'ocean_modelling_data',
                     'melt_warm_target_term3_v2.nc'))
    t3_mean = warm.melt_rate - cold.melt_rate
    t3_sigma = np.sqrt(warm.melt_rate_uncert**2 + cold.melt_rate_uncert**2)

    t4 = xr.load_dataset(
        os.path.join(param, 'ocean_observations_data',
                     'melt_observations_target_term4.nc'))

    return dict(t1_mean=t1_mean, t1_sigma=t1_sigma,
                t2_mean=buttressing['melt_mean'],
                t2_sigma=buttressing['melt_mean_err'],
                t2_weights=t2_weights,
                t3_mean=t3_mean, t3_sigma=t3_sigma,
                t4_mean=t4.melt_rate, t4_sigma=t4.melt_rate_uncert)
