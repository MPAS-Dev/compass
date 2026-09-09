"""
Tests for the ISMIP7 ocean-state and dataset registry.

These check the registry's internal consistency, which is what keeps the
8 km replication and the MALI-mesh calibration driving the same states.  They
do not touch the ISMIP7 datasets themselves, which are not in the repository.
"""

import pytest

from compass.landice.tests.ismip7_calibration import datasets

BASE_PATH = '/not/a/real/path'


def test_subset_sizes_match_the_protocol():
    """
    7 states for the minimal set, 11 for the recommended one and 28 for
    everything ISMIP7 distributes.
    """
    assert len(datasets.ocean_states(BASE_PATH, 'minimal')) == 7
    assert len(datasets.ocean_states(BASE_PATH, 'recommended')) == 11
    assert len(datasets.ocean_states(BASE_PATH, 'all')) == 28


def test_subsets_are_nested():
    """A larger subset must contain everything a smaller one does."""
    minimal = {state.name for state in
               datasets.ocean_states(BASE_PATH, 'minimal')}
    recommended = {state.name for state in
                   datasets.ocean_states(BASE_PATH, 'recommended')}
    every = {state.name for state in datasets.ocean_states(BASE_PATH, 'all')}

    assert minimal <= recommended <= every


def test_an_unknown_subset_is_rejected():
    with pytest.raises(ValueError, match="must be 'minimal'"):
        datasets.ocean_states(BASE_PATH, 'everything')


def test_state_names_are_unique():
    """Names become directory names, so a clash would silently overwrite."""
    names = [state.name for state in datasets.ocean_states(BASE_PATH, 'all')]

    assert len(names) == len(set(names))


def test_every_state_feeds_a_term():
    """A state that constrains nothing would just waste a MALI run."""
    for state in datasets.ocean_states(BASE_PATH, 'all'):
        assert state.term in ('J1,J2', 'J3', 'J4')


def test_the_climatology_is_the_only_present_day_state():
    """J1 and J2 are present-day terms, driven by the one climatology."""
    states = datasets.ocean_states(BASE_PATH, 'all')
    climatology = [state for state in states if state.kind == 'climatology']

    assert len(climatology) == 1
    assert climatology[0].name == 'climatology'
    assert climatology[0].term == 'J1,J2'


def test_ocean_models_come_in_cold_and_warm_pairs():
    """J3 is a warm-minus-cold difference, so both must be present."""
    states = [state for state in datasets.ocean_states(BASE_PATH, 'all')
              if state.kind == 'model']
    by_label = {}
    for state in states:
        by_label.setdefault(state.label, set()).add(state.state)

    assert len(by_label) == 7
    for label, which in by_label.items():
        assert which == {'cold', 'warm'}, label


def test_regional_models_declare_the_basins_they_cover():
    """
    Melt outside a regional model's domain must be discarded, so the
    covered basins have to be recorded.  Basin numbers are the ISMIP7
    0-based convention.
    """
    states = {state.label: state
              for state in datasets.ocean_states(BASE_PATH, 'all')
              if state.kind == 'model'}

    assert states['mathiot'].basins is None
    assert states['naughten_ais_1'].basins is None
    assert states['jourdain_naughten'].basins == (9, 14)
    assert states['naughten_naughten'].basins == (9, 14)
    assert states['timmermann'].basins == (14,)


def test_observations_constrain_the_eastern_amundsen():
    """
    PIG and Dotson are both in ISMIP7 basin 9, the Eastern Amundsen, in the
    0-based convention the protocol uses.
    """
    states = [state for state in datasets.ocean_states(BASE_PATH, 'all')
              if state.kind == 'obs']

    assert len(states) == 13
    for state in states:
        assert state.basins == (9,)
        assert state.year in datasets.OBS_YEARS


def test_the_published_weighting_is_a_strict_subset():
    """
    The published J4 weighting is PIG in 2009 and 2012 only -- 2 of the 18
    available observations.  Pinning it here guards the replication.
    """
    assert datasets.PUBLISHED_T4_REGIONS == ('pig',)
    assert datasets.PUBLISHED_T4_YEARS == (2009, 2012)
    assert set(datasets.PUBLISHED_T4_YEARS) < set(datasets.OBS_YEARS)
    assert len(datasets.PUBLISHED_T3_MODELS) == 4


def test_mask_file_names_follow_the_resolution():
    """The ISMIP masks are distributed at several resolutions."""
    files = datasets.mask_files(BASE_PATH, resolution_km=8)

    assert files['basins'].endswith('basin_numbers_ismip8km_v2.nc')
    assert files['bfrn'].endswith('BFRN_ismip8km_v2.nc')
    assert files['floating'].endswith('floatingmask_ismip8km.nc')
    # the shelf mask is only distributed at 8 km
    assert files['shelves'].endswith('shelf_mask_ismip8km.nc')


def test_shelf_ids_and_the_pine_island_cut():
    """
    The worked example keeps only Pine Island's main trunk, cutting it at a
    fixed x.  Getting the ids or the cut wrong would silently change J4.
    """
    assert datasets.PIG_ID == 110
    assert datasets.DOTSON_ID == 97
    assert datasets.PIG_X_MAX == -1.625e6


def test_missing_files_reports_absent_inputs():
    """The registry points at a fake root, so everything is missing."""
    states = datasets.ocean_states(BASE_PATH, 'minimal')

    missing = datasets.missing_files(states)

    # both the thermal-forcing and the salinity file of each state
    assert len(missing) == 2 * len(states)
