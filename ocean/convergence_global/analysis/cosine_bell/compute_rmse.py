#!/usr/bin/env python

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dpi = 200


def rmse(resTag):
    """
    resTag is the resolution to compute RMSE
    """

    fid = open('../../../{}/cosine_bell/initial_state/namelist.ocean'.format(resTag),
               'r')
    temp = fid.readlines()

    fid.close()
    for i, line in enumerate(temp):
        if "config_cosine_bell_lat_center" in line:
            ii = line.find('=')+1
            latCent = float(line[ii:])
        if "config_cosine_bell_lon_center" in line:
            ii = line.find('=')+1
            lonCent = float(line[ii:])
        if "config_cosine_bell_radius" in line:
            ii = line.find('=')+1
            radius = float(line[ii:])
        if "config_cosine_bell_psi0" in line:
            ii = line.find('=')+1
            psi0 = float(line[ii:])
        if "config_cosine_bell_vel_pd" in line:
            ii = line.find('=')+1
            pd = float(line[ii:])

    init = xr.open_dataset('../../../{}/cosine_bell/initial_state/initial_state.nc'
                           ''.format(resTag))
    # find time since the beginning of run
    ds = xr.open_dataset('../../../{}/cosine_bell/forward/output/'
                         'output.0001-01-01_00.00.00.nc'.format(resTag))
    for j in range(len(ds.xtime)):
        tt = str(ds.xtime[j].values)
        tt.rfind('_')
        DY = float(tt[10:12])-1
        if DY == pd:
            sliceTime = j
            break
    HR = float(tt[13:15])
    MN = float(tt[16:18])
    t = 86400.0*DY+HR*3600.+MN
    # find new location of blob center
    # center is based on equatorial velocity
    R = init.sphere_radius
    distTrav = 2.0*3.14159265*R / (86400.0*pd)*t
    # distance in radians is
    distRad = distTrav / R
    newLon = lonCent + distRad
    if newLon > 2.0*np.pi:
        newLon -= 2.0*np.pi

    # construct analytic tracer
    tracer = np.zeros_like(init.tracer1[0, :, 0].values)
    latC = init.latCell.values
    lonC = init.lonCell.values
    temp = R*np.arccos(np.sin(latCent)*np.sin(latC) +
                       np.cos(latCent)*np.cos(latC)*np.cos(lonC - newLon))
    mask = temp < radius
    tracer[mask] = psi0 / 2.0 * (1.0 + np.cos(3.1415926*temp[mask]/radius))

    # oad forward mode data
    tracerF = ds.tracer1[sliceTime, :, 0].values
    rmseValue = np.sqrt(np.mean((tracerF-tracer)**2))

    init.close()
    ds.close()
    return rmseValue, init.dims['nCells']


res = ['QU60', 'QU90', 'QU120', 'QU150', 'QU180', 'QU210', 'QU240']
xtemp = []
ytemp = []
for i in range(len(res)):
    exec('rmse'+res[i]+',nCells'+res[i]+' = rmse(res[i])')
    exec('xtemp.append(nCells'+res[i]+')')
    exec('ytemp.append(rmse'+res[i]+')')
xdata = np.asarray(xtemp)
ydata = np.asarray(ytemp)

p = np.polyfit(np.log10(xdata), np.log10(ydata), 1)
conv = abs(p[0])*2.0

yfit = xdata**p[0]*10**p[1]

plt.loglog(xdata, yfit, 'k')
plt.loglog(xdata, ydata, 'or')
plt.annotate('Order of Convergence = {}'.format(np.round(conv, 3)),
             xycoords='axes fraction', xy=(0.3, 0.95), fontsize=14)
plt.xlabel('Number of Grid Cells', fontsize=14)
plt.ylabel('L2 Norm', fontsize=14)
plt.savefig('convergence.png', bbox_inches='tight', pad_inches=0.1)

