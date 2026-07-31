import cmocean
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from inpoly import inpoly2


def plot_subgrid(ax, X, Y, bathy, solution, mesh, shape, ssh, points):

    print(mesh.dims)
    xVertex = mesh['xVertex'].values
    yVertex = mesh['yVertex'].values
    verticesOnCell = mesh['verticesOnCell'].values - 1
    nEdgesOnCell = mesh['nEdgesOnCell'].values

    tsnap = 227
    h0 = 0.0

    pts = np.vstack((X.ravel(), Y.ravel())).T
    for iCell in range(mesh.dims['nCells']):
        print(iCell)
        xnode = xVertex[verticesOnCell[iCell, 0:nEdgesOnCell[iCell]]]
        ynode = yVertex[verticesOnCell[iCell, 0:nEdgesOnCell[iCell]]]
        nodes = np.vstack((xnode, ynode)).T
        edges = []
        for i in range(nEdgesOnCell[iCell]):
            edges.append([i, (i + 1) % (nEdgesOnCell[iCell])])
        edges = np.array(edges)

        inpt, onpt = inpoly2(pts, nodes, edges)

        inwet = np.where(
            inpt & ((bathy + h0) < ssh[tsnap, iCell]) &
            (ssh[tsnap, iCell] < 2.0))

        solution[inwet] = ssh[tsnap, iCell]

    sol_min = -2
    sol_max = 2
    levels = np.linspace(sol_min, sol_max, 50)
    cmap = plt.get_cmap('RdBu')
    minval = 0.2
    maxval = 0.8
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'truncated RdBu', cmap(np.linspace(minval, maxval, 256)))
    ax.set_facecolor(color='lightgrey')
    c = ax.contourf(
        X, Y, solution.reshape(shape), vmin=sol_min, vmax=sol_max,
        levels=levels, cmap=cmap, extend='both')

    ax.set_aspect('equal', 'box')
    ax.set_title(f'{res}m resolution')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')

    formatter = ticker.FuncFormatter(lambda x_val, pos: f'{x_val / 1000:g}')

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_aspect('equal', 'box')
    ax.scatter(
        points[:, 0], points[:, 1], 15, 'k')
    ax.set_xlim([0.0, 4000])
    ax.set_ylim([0.0, 3500])

    return c


mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


ds_dem = xr.open_dataset('buttermilk_bathy.nc')
bathy = ds_dem['Band1'].values
x = ds_dem['x'].values
y = ds_dem['y'].values
X, Y = np.meshgrid(x, y)
solution = np.full_like(bathy, np.nan)
shape = bathy.shape

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)
vmin = -20
vmax = 20
levels = np.linspace(vmin, vmax, 100)
print(levels)
c = ax.contourf(
    X, Y, bathy, cmap=cmocean.cm.topo, vmin=vmin, vmax=vmax,
    levels=levels, extend='both')
tick_step = 5
ticks = np.arange(vmin, vmax + tick_step, tick_step)
cb = fig.colorbar(c, ticks=ticks)
ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
cb.set_label('bathymetry/topography (m)')
ax.set_aspect('equal', 'box')
fig.tight_layout()
fig.savefig('bathy.png', bbox_inches='tight', dpi=400)

# raise SystemExit(0)

resolutions = [256, 128, 64]
# resolutions = [256]
# points = 1000 * np.array(
#     [[2.8, 0.53], [1.9, 1.66], [2.4, 3.029], [2.51, 3.027], [1.26, 1.56]])
points = 1000 * np.array(
    [[2.8, 0.53], [1.9, 1.66], [3.0, 3.25], [2.6, 3.25], [1.26, 1.56]])
ncols = len(resolutions)
if ncols < 2:
    ncols = 2
fig, ax = plt.subplots(
    nrows=1, ncols=ncols, figsize=(3 * ncols, 3), constrained_layout=True)
for i, res in enumerate(resolutions):
    ds_mesh = xr.open_dataset(f'output_{res}m.nc')
    ssh = ds_mesh['ssh'].values
    cm = plot_subgrid(
        ax[i], X, Y, bathy.ravel(), solution.ravel(),
        ds_mesh, shape, ssh, points)

tick_step = 0.5
sol_min = -2
sol_max = 2
ticks = np.arange(sol_min, sol_max + tick_step, tick_step)
cb = fig.colorbar(cm, ax=ax[-1], shrink=0.7, ticks=ticks, extend='both')
title = 'b)'
ha = 'left'
x = 0.0
y = 0.98
for j, sta in enumerate(range(len(points))):
    if (j == 2) | (j == 3):
        xoffset = 50
        yoffset = -200
    else:
        xoffset = 0
        yoffset = 100
    ax[0].text(
        points[j, 0] + xoffset, points[j, 1] + yoffset, str(j + 1), color='k')

fig.suptitle(title, ha=ha, x=x, y=y, fontsize='x-large')
cb.set_label('ssh (m)')
fig.savefig('subgrid_on_subgrid.png', dpi=400)
