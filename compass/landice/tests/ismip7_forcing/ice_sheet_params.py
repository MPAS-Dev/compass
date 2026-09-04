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
        'ocean_grid': 'ocean',
        'ocean_3d': True,
        'ocean_temporal': 'decade',
        'atm_model': None,
        'ocean_model': None,
    },
    'gis': {
        'projection': 'gis-bamber',
        'prefix': 'GrIS',
        'atm_resolution': '1000m',
        'atm_version': 'v2',
        'ocean_version': 'v2',
        'ocean_grid': 'ocean',
        'ocean_3d': False,
        'ocean_temporal': 'yearly',
        'atm_model': None,
        'ocean_model': None,
    },
}

# Overrides applied for the OCX (reanalysis) scenario. OCX has no distinct
# ESM model: it uses fixed reanalysis products (RACMO for the atmosphere and
# EN4 for the ocean) at data version v1. When scenario is 'OCX' the [ismip7]
# model option is ignored and these sources are used instead.
_OCX_OVERRIDES = {
    'gis': {
        'atm_version': 'v1',
        'ocean_version': 'v1',
        'ocean_grid': 'ocean-1000m',
        'atm_model': 'RACMO2.3p2-ERA',
        'ocean_model': 'EN4',
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
    params = dict(_PARAMS[ice_sheet])

    scenario = config.get("ismip7", "scenario")
    if scenario == "OCX":
        if ice_sheet not in _OCX_OVERRIDES:
            raise ValueError(
                f"The OCX scenario is not yet supported for ice_sheet "
                f"'{ice_sheet}'.")
        params.update(_OCX_OVERRIDES[ice_sheet])

    return params
