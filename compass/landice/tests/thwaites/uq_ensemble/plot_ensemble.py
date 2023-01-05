#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import netCDF4
import glob
import sys, os

rhoi = 910.0
rhosw = 1028.0

runs = sorted(glob.glob("run*"))
nRuns = len(runs)

# initialize param vectors
vmThresh = np.zeros((nRuns,)) * np.nan
vmSpdLim = np.zeros((nRuns,)) * np.nan
# initialize QOIs
SLRAll = np.zeros((nRuns,)) * np.nan
grdAreaLossAll = np.zeros((nRuns,)) * np.nan
areaLossAll = np.zeros((nRuns,)) * np.nan
fltAreaLossAll = np.zeros((nRuns,)) * np.nan

def get_nl_option(file, option_name):
    with open(file, "r") as fp:
        for line in fp:
            if option_name in line:
                fp.close()
                return float(line.split("=")[1].strip())



for idx, run in enumerate(runs):
    # get param valuse for this run
    # (get from namelist in case run didn't produce output)
    nlFile = run + "/namelist.landice"
    vmThresh[idx] = get_nl_option(nlFile, "config_floating_von_Mises_threshold_stress") / 1000.0  # convert to kPa
    vmSpdLim[idx] = get_nl_option(nlFile, "config_calving_speed_limit") * 3600.0 * 24.0 * 365.0 / 1000.0  # convert to km/yr
    

    fpath = run + "/globalStats.nc"
    if os.path.exists(fpath):
        f = netCDF4.Dataset(fpath, 'r')
        years = f.variables['daysSinceStart'][:] / 365.0
        # Only process runs that have completed
        if years[-1] == 100.0:
            VAF = f.variables['volumeAboveFloatation'][:]
            SLR = (VAF[0] - VAF) / 3.62e14 * rhoi / rhosw * 1000.
            SLRAll[idx] = SLR[-1]

            grdArea = f.variables['groundedIceArea'][:] / 1000.0**2  # in km^2
            grdAreaLossAll[idx] = grdArea[0] - grdArea[-1]

            fltArea = f.variables['floatingIceArea'][:] / 1000.0**2  # in km^2
            fltAreaLossAll[idx] = fltArea[0] - fltArea[-1]

            iceArea = f.variables['totalIceArea'][:] / 1000.0**2  # in km^2
            areaLossAll[idx] = iceArea[0] - iceArea[-1]

        f.close()

# plot results
fig = plt.figure(1, figsize=(14, 11), facecolor='w')
nrow=2
ncol=2

axSLR = fig.add_subplot(nrow, ncol, 1)
plt.title('SLR at 2100 (mm)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('calving speed limit (km/yr)')
plt.grid()
plt.scatter(vmThresh, vmSpdLim, s=50, c=SLRAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(SLRAll))[0]
plt.plot(vmThresh[badIdx], vmSpdLim[badIdx], 'kx')
plt.colorbar()

axArea = fig.add_subplot(nrow, ncol, 2)
plt.title('Total area loss at 2100 (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('calving speed limit (km/yr)')
plt.grid()
plt.scatter(vmThresh, vmSpdLim, s=50, c=areaLossAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(areaLossAll))[0]
plt.plot(vmThresh[badIdx], vmSpdLim[badIdx], 'kx')
plt.colorbar()

axGrdArea = fig.add_subplot(nrow, ncol, 3)
plt.title('Grounded area loss at 2100 (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('calving speed limit (km/yr)')
plt.grid()
plt.scatter(vmThresh, vmSpdLim, s=50, c=grdAreaLossAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(grdAreaLossAll))[0]
plt.plot(vmThresh[badIdx], vmSpdLim[badIdx], 'kx')
plt.colorbar()

axfltArea = fig.add_subplot(nrow, ncol, 4)
plt.title('Floating area loss at 2100 (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('calving speed limit (km/yr)')
plt.grid()
plt.scatter(vmThresh, vmSpdLim, s=50, c=fltAreaLossAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(fltAreaLossAll))[0]
plt.plot(vmThresh[badIdx], vmSpdLim[badIdx], 'kx')
plt.colorbar()

fig.tight_layout()
plt.show()






