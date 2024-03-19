
# -- long_name strings for NetCDF variables

class base:
    pass


names = base()
names.bed_elevation = "elevation of bed"
names.ocn_thickness = "thickness of ocn"
names.ice_thickness = "thickness of ice"
names.ocn_cover = "fractional cover of ocn, 0-1"
names.ice_cover = "fractional cover of ice, 0-1"
names.bed_slope = "RMS magnitude of bed slopes"
names.bed_slope_deg = "arc-tangent of RMS bed slopes"
names.bed_dz_dx = "derivative of bed elevation along lon.-axis"
names.bed_dz_dy = "derivative of bed elevation along lat.-axis"
names.bed_elevation_profile = "sub-grid percentiles of bed elevation"
names.bed_slope_profile = "sub-grid percentiles of RMS bed slope"
names.ocn_thickness_profile = "sub-grid percentiles of ocn thickness"
names.ice_thickness_profile = "sub-grid percentiles of ice thickness"
