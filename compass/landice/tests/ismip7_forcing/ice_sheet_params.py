"""
Ice-sheet-specific parameters for ISMIP7 forcing data processing.
"""

# Parameters that differ between AIS and GrIS
_PARAMS = {
    'ais': {
        'projection': 'ais-bedmap2',
        'prefix': 'AIS',
        'atm_resolution': '2000m',
        'atm_version': 'v2',
        'ocean_version': 'v3',
        'ocean_3d': True,
        'ocean_temporal': 'decade',
    },
    'gis': {
        'projection': 'gis-bamber',
        'prefix': 'GrIS',
        'atm_resolution': '1000m',
        'atm_version': 'v2',
        'ocean_version': 'v2',
        'ocean_3d': False,
        'ocean_temporal': 'yearly',
    },
}


def get_params(config):
    """
    Get ice-sheet-specific parameters from the config.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    Returns
    -------
    params : dict
        Dictionary of ice-sheet-specific parameters
    """
    ice_sheet = config.get("ismip7", "ice_sheet")
    if ice_sheet not in _PARAMS:
        raise ValueError(
            f"Unknown ice_sheet '{ice_sheet}'. "
            f"Must be one of: {list(_PARAMS.keys())}")
    return _PARAMS[ice_sheet]
