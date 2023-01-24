#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import netCDF4
import glob
import sys, os
import yaml

targetYear = 100.0  # model year from start at which to calculate statistics
labelRuns = True

rhoi = 910.0
rhosw = 1028.0

runs = sorted(glob.glob("run*"))
nRuns = len(runs)

# initialize param vectors
vmThresh = np.zeros((nRuns,)) * np.nan
vmSpdLim = np.zeros((nRuns,)) * np.nan
fricExp = np.zeros((nRuns,)) * np.nan
# initialize QOIs
SLRAll = np.zeros((nRuns,)) * np.nan
grdAreaChangeAll = np.zeros((nRuns,)) * np.nan
areaChangeAll = np.zeros((nRuns,)) * np.nan
fltAreaChangeAll = np.zeros((nRuns,)) * np.nan

def get_nl_option(file, option_name):
    with open(file, "r") as fp:
        for line in fp:
            if option_name in line:
                fp.close()
                return float(line.split("=")[1].strip())

for idx, run in enumerate(runs):
    # get param values for this run
    # Could read from param vec file, but that leaves room for error
    # Better to get directly from run files

    # Get values from namelist in case run didn't produce output.nc files.
    nlFile = run + "/namelist.landice"
    vmThresh[idx] = get_nl_option(nlFile, "config_floating_von_Mises_threshold_stress") / 1000.0  # convert to kPa
    #vmSpdLim[idx] = get_nl_option(nlFile, "config_calving_speed_limit") * 3600.0 * 24.0 * 365.0 / 1000.0  # convert to km/yr

    # read yaml file for fric exp
    with open(run + '/albany_input.yaml', 'r') as stream:
        try:
            loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    fricExp[idx] = loaded['ANONYMOUS']['Problem']['LandIce BCs']['BC 0']['Basal Friction Coefficient']['Power Exponent']

    fpath = run + "/globalStats.nc"
    if os.path.exists(fpath):
        f = netCDF4.Dataset(fpath, 'r')
        years = f.variables['daysSinceStart'][:] / 365.0
        # Only process runs that have reached target year
        indices = np.nonzero(years >= targetYear)[0]
        if len(indices) > 0:
            ii = indices[0]
            print(f'{run} using year {years[ii]}')
            VAF = f.variables['volumeAboveFloatation'][:]
            SLR = (VAF[0] - VAF) / 3.62e14 * rhoi / rhosw * 1000.
            SLRAll[idx] = SLR[ii]

            grdArea = f.variables['groundedIceArea'][:] / 1000.0**2  # in km^2
            grdAreaChangeAll[idx] = grdArea[ii] - grdArea[0]

            fltArea = f.variables['floatingIceArea'][:] / 1000.0**2  # in km^2
            fltAreaChangeAll[idx] = fltArea[ii] - fltArea[0]

            iceArea = f.variables['totalIceArea'][:] / 1000.0**2  # in km^2
            areaChangeAll[idx] = iceArea[ii] - iceArea[0]

        f.close()

# plot results
fig = plt.figure(1, figsize=(14, 11), facecolor='w')
nrow=2
ncol=2

axSLR = fig.add_subplot(nrow, ncol, 1)
plt.title(f'SLR at year {targetYear} (mm)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=50, c=SLRAll, plotnonfinite=False)
if labelRuns:
    for i in range(nRuns):
        plt.annotate(f'{runs[i][3:]}', (vmThresh[i], fricExp[i]))
badIdx = np.nonzero(np.isnan(SLRAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

axArea = fig.add_subplot(nrow, ncol, 2)
plt.title(f'Total area change at year {targetYear} (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=50, c=areaChangeAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(areaChangeAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

axGrdArea = fig.add_subplot(nrow, ncol, 3)
plt.title(f'Grounded area change at year {targetYear} (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=50, c=grdAreaChangeAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(grdAreaChangeAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

axfltArea = fig.add_subplot(nrow, ncol, 4)
plt.title(f'Floating area change at year {targetYear} (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=50, c=fltAreaChangeAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(fltAreaChangeAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

fig.tight_layout()
plt.show()
