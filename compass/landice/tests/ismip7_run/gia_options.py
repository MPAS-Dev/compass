"""
Shared parsing for the ``gia_model`` config option, which selects between
the two available (mutually exclusive) glacial isostatic adjustment (GIA)
couplings for the ISMIP7 test cases. Lives at the ``ismip7_run`` level
(rather than inside ``ismip7_ais``) so both the AIS and GrIS test cases can
use it symmetrically, as siblings importing from their shared parent
package, without either reaching into the other's internals. Kept in its
own module (rather than in ``ismip7_run/__init__.py`` or either test
case's own ``__init__.py``) so it can be imported by step modules without
creating a circular import with those packages' ``__init__.py`` files.
"""

GIA_MODEL_OPTIONS = ('none', '1dSLM', 'FastIsostasy')


def parse_gia_model(config, section='ismip7_run_ais'):
    """
    Parse the ``gia_model`` config option into ``sea_level_model`` and
    ``fastisostasy`` flags.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        The config options

    section : str, optional
        The config section to read ``gia_model`` from -- ``ismip7_run_ais``
        (the default) or ``ismip7_run_gris``

    Returns
    -------
    sea_level_model : bool
        Whether the 1-D sea-level model GIA coupling is enabled

    fastisostasy : bool
        Whether the FastIsostasy GIA coupling is enabled
    """
    gia_model = config.get(section, 'gia_model')
    valid = {opt.lower(): opt for opt in GIA_MODEL_OPTIONS}
    if gia_model.lower() not in valid:
        raise ValueError(
            f"Unknown gia_model '{gia_model}'. Valid options are: "
            f"{', '.join(GIA_MODEL_OPTIONS)}")
    canonical = valid[gia_model.lower()]
    sea_level_model = canonical == '1dSLM'
    fastisostasy = canonical == 'FastIsostasy'
    return sea_level_model, fastisostasy
