"""
Tests for the ISMIP7 mask checks.

The basin numbering is the most dangerous thing to get wrong here: ISMIP7
is 0-based and MALI is 1-based, so confusing them mis-assigns every basin
while still producing plausible-looking aggregates.  These test the checks
that catch it.
"""

import logging

import numpy as np
import pytest
import xarray as xr

from compass.landice.tests.ismip7_calibration.ais.remap_masks import (
    EASTERN_AMUNDSEN_BASIN,
    REGION_CODES,
    _check_shelves_are_in_their_basin,
)


def _logger():
    logger = logging.getLogger('test_remap_masks')
    logger.addHandler(logging.NullHandler())
    return logger


def _masks(basins, regions):
    ds = xr.Dataset()
    ds['ismip7BasinNumber'] = ('nCells', np.asarray(basins, dtype=np.int32))
    ds['ismip7ShelfRegion'] = ('nCells', np.asarray(regions, dtype=np.int32))
    return ds


def test_shelves_in_the_eastern_amundsen_pass():
    """PIG and Dotson both drain into ISMIP7 basin 9."""
    ds = _masks(basins=[EASTERN_AMUNDSEN_BASIN] * 4 + [3],
                regions=[REGION_CODES['pig'], REGION_CODES['pig'],
                         REGION_CODES['dotson'], REGION_CODES['dotson'],
                         REGION_CODES['none']])

    _check_shelves_are_in_their_basin(ds, _logger())


def test_off_by_one_basin_numbering_is_caught():
    """
    Using MALI's 1-based numbering where ISMIP7's 0-based is expected puts
    the shelves in basin 10 instead of 9.  That is the mistake this check
    exists for.
    """
    ds = _masks(basins=[EASTERN_AMUNDSEN_BASIN + 1] * 4,
                regions=[REGION_CODES['pig']] * 2 +
                        [REGION_CODES['dotson']] * 2)

    with pytest.raises(ValueError, match='numbering is probably off'):
        _check_shelves_are_in_their_basin(ds, _logger())


def test_a_few_stray_cells_are_tolerated():
    """
    Nearest-neighbour remapping can put a cell or two on a basin boundary
    into a neighbour, which is not an error.
    """
    basins = [EASTERN_AMUNDSEN_BASIN] * 19 + [8]
    ds = _masks(basins=basins, regions=[REGION_CODES['pig']] * 20)

    _check_shelves_are_in_their_basin(ds, _logger())


def test_a_mesh_with_no_shelf_cells_is_an_error():
    """
    If the shelf mask misses the mesh entirely, term J4 would have nothing
    to aggregate over and would fail later and less clearly.
    """
    ds = _masks(basins=[1, 2, 3], regions=[REGION_CODES['none']] * 3)

    with pytest.raises(ValueError, match='nothing to aggregate'):
        _check_shelves_are_in_their_basin(ds, _logger())
