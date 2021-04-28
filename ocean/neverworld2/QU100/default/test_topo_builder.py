import cartopy
import matplotlib.pyplot as plt
import numpy
import scipy.interpolate
import topo_builder

# NeverWorld2 domain
NW2_lonW, NW2_lonE = 0, 80
NW2_latS, NW2_latN = -70, 70

D0 = 4000 # Nominal depth (m)
cd = 200 # Depth of coastal shelf (m)
drake = 2500 # Depth of Drake sill (m)
cw = 5 # Width of coastal shelf (degrees)

# Logical domain (grid points)
nj, ni = 140, 80


# Simple "Atlantic" box with re-entrant Drake passage

T = topo_builder.topo(nj, ni, dlon=NW2_lonE, dlat=NW2_latN-NW2_latS, lat0=NW2_latS, D=D0)
T.add_NS_coast(NW2_lonW, -40, 90, cw, cd)
T.add_NS_coast(NW2_lonE, -40, 90, cw, cd)
T.add_NS_coast(NW2_lonW, -90, -60, cw, cd)
T.add_NS_coast(NW2_lonE, -90, -60, cw, cd)
T.add_EW_coast(-360, 360, NW2_latS, cw, cd)
T.add_EW_coast(-360, 360, NW2_latN, cw, cd)

fig = plt.figure(figsize=(12,10))
T.plot(fig, Atlantic_lon_offset=-84)
plt.savefig('test_topo.png')
