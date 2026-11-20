from mpas_tools.cime.constants import constants as _mpas_tools_constants

# E3SM's shared constants (shr_const_mod) now come from the Physical
# Constants Dictionary (PCD), but those in mpas_tools haven't caught up yet
# (MPAS-Dev/MPAS-Tools#781).  Gravity is corrected here because MPAS-Ocean,
# MALI and E3SM's coupler all use the PCD value, so ice-shelf pressures
# computed by compass must too.
constants = dict(_mpas_tools_constants)
constants['SHR_CONST_G'] = 9.80665
