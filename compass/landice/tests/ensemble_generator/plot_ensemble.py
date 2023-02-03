#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import netCDF4
import glob
import sys, os
import yaml
import xarray as xr
import matplotlib.tri as tri

targetYear = 20.0  # model year from start at which to calculate statistics
labelRuns = True

rhoi = 910.0
rhosw = 1028.0

runs = sorted(glob.glob("run*"))
nRuns = len(runs)


# obs from Rignot et al 2019 PNAS
obs_discharge_thwaites = np.array([82.42,	83.25,	84.08,	84.90,	85.73,	86.56,	87.39,	88.22,	89.04,	89.87,	90.70,	91.40,	92.10,	92.80,	93.50,	94.20,	94.90,	95.60,	95.90,	96.20,	96.50,	96.80,	97.10,	97.40,	97.75,	98.10,	101.60,	101.40,	102.80,	104.30,	109.70,	114.10,	115.80,	118.70,	121.60,	123.80,	117.20,	115.10,	119.68])
obs_discharge_haynes = np.array([10.10, 10.25, 10.40, 10.55, 10.70, 10.85, 11.00, 11.15, 11.30, 11.45, 11.60, 11.49, 11.37, 11.26, 11.14, 11.03, 10.91, 10.80, 10.93, 11.07, 11.20, 11.33, 11.47, 11.60, 11.35, 11.10, 12.20, 11.90, 12.50, 12.11, 12.80, 12.80, 12.90, 12.80, 13.90, 14.70, 12.30, 11.60, 12.53])
obs_discharge = obs_discharge_thwaites + obs_discharge_haynes
obs_years = np.arange(1979, 2017+1) - 2000.0
obs_sigmaD = (3.93**2 + 1.00**2)**0.5  # thwaites + haynes

# initialize param vectors
vmThresh = np.zeros((nRuns,)) * np.nan
vmSpdLim = np.zeros((nRuns,)) * np.nan
fricExp = np.zeros((nRuns,)) * np.nan
# initialize QOIs
SLRAll = np.zeros((nRuns,)) * np.nan
grdAreaChangeAll = np.zeros((nRuns,)) * np.nan
areaChangeAll = np.zeros((nRuns,)) * np.nan
fltAreaChangeAll = np.zeros((nRuns,)) * np.nan
GLfluxAll = np.zeros((nRuns,)) * np.nan

def get_nl_option(file, option_name):
    with open(file, "r") as fp:
        for line in fp:
            if option_name in line:
                fp.close()
                return float(line.split("=")[1].strip())

# time series plot
figTS = plt.figure(2, figsize=(8, 12), facecolor='w')
nrow=2
ncol=1
axSLRts = figTS.add_subplot(nrow, ncol, 1)
#plt.title(f'SLR at year {targetYear} (mm)')
plt.xlabel('Year')
plt.ylabel('SLR contribution (mm)')
plt.grid()

axGLFts = figTS.add_subplot(nrow, ncol, 2)
plt.xlabel('Year')
plt.ylabel('GL flux (Gt)')
plt.grid()
axGLFts.fill_between(obs_years, obs_discharge - 2.0*obs_sigmaD, obs_discharge + 2.0*obs_sigmaD, color='k', alpha=0.1, label='D obs')

# maps
figMaps = plt.figure(3, figsize=(8, 12), facecolor='w')
nrow=2
ncol=1
axMaps = figMaps.add_subplot(nrow, ncol, 1)
axMaps.axis('equal')


for idx, run in enumerate(runs):
    print(f'Analyzing {run}')
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

    fpath = run + "/output/globalStats.nc"
    if os.path.exists(fpath):
        f = netCDF4.Dataset(fpath, 'r')
        years = f.variables['daysSinceStart'][:] / 365.0

        VAF = f.variables['volumeAboveFloatation'][:]
        SLR = (VAF[0] - VAF) / 3.62e14 * rhoi / rhosw * 1000.
        groundingLineFlux = f.variables['groundingLineFlux'][:] / 1.0e12  # in Gt


        # plot time series
        axSLRts.plot(years, SLR)
        axGLFts.plot(years[1:], groundingLineFlux[1:])  # ignore first entry which is 0

        # Only process runs that have reached target year
        indices = np.nonzero(years >= targetYear)[0]
        if len(indices) > 0:
            ii = indices[0]
            print(f'{run} using year {years[ii]}')
            SLRAll[idx] = SLR[ii]

            grdArea = f.variables['groundedIceArea'][:] / 1000.0**2  # in km^2
            grdAreaChangeAll[idx] = grdArea[ii] - grdArea[0]

            fltArea = f.variables['floatingIceArea'][:] / 1000.0**2  # in km^2
            fltAreaChangeAll[idx] = fltArea[ii] - fltArea[0]

            iceArea = f.variables['totalIceArea'][:] / 1000.0**2  # in km^2
            areaChangeAll[idx] = iceArea[ii] - iceArea[0]

            GLfluxAll[idx] = groundingLineFlux[ii]

        # plot map
        DS = xr.open_mfdataset(run + '/output/' + 'output_*.nc', combine='nested', concat_dim='Time', decode_timedelta=False, chunks={"Time": 10})
        yearsOutput = DS['daysSinceStart'].values[:] / 365.0
        indices = np.nonzero(yearsOutput >= targetYear)[0]
        if len(indices) > 0:
            ii = indices[0]

            thickness = DS['thickness'].values
            bedTopo = DS['bedTopography'].values
            xCell = DS['xCell'].values[0,:]
            yCell = DS['yCell'].values[0,:]

            triang = tri.Triangulation(xCell, yCell)


            axMaps.tricontour(triang, thickness[0], [1.0], colors='k')
            axMaps.tricontour(triang, thickness[ii], [1.0], colors='r')

            ii5 = np.nonzero(yearsOutput >= 10.0)[0][0]

            grd0 = ((thickness[0,:]*910.0/1028.0+bedTopo[0,:])>0.0)*(thickness[0,:]>0.0)
            grd5 = ((thickness[ii5,:]*910.0/1028.0+bedTopo[ii5,:])>0.0)*(thickness[ii5,:]>0.0)
            grd = ((thickness[ii,:]*910.0/1028.0+bedTopo[ii,:])>0.0)*(thickness[ii,:]>0.0)
            axMaps.tricontour(triang, grd0, [0.5], colors='k')
            axMaps.tricontour(triang, grd5, [0.5], colors='c')
            axMaps.tricontour(triang, grd, [0.5], colors='b')




        f.close()

# plot results

markerSize = 100

fig = plt.figure(1, figsize=(14, 11), facecolor='w')
nrow=2
ncol=2

axSLR = fig.add_subplot(nrow, ncol, 1)
plt.title(f'SLR at year {targetYear} (mm)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=markerSize, c=SLRAll, plotnonfinite=False)
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
plt.scatter(vmThresh, fricExp, s=markerSize, c=areaChangeAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(areaChangeAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

axGrdArea = fig.add_subplot(nrow, ncol, 3)
plt.title(f'Grounded area change at year {targetYear} (km$^2$)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=markerSize, c=grdAreaChangeAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(grdAreaChangeAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()

#axfltArea = fig.add_subplot(nrow, ncol, 4)
#plt.title(f'Floating area change at year {targetYear} (km$^2$)')
#plt.xlabel('von Mises stress threshold (kPa)')
#plt.ylabel('basal friction law exponent')
#plt.grid()
#plt.scatter(vmThresh, fricExp, s=markerSize, c=fltAreaChangeAll, plotnonfinite=False)
#badIdx = np.nonzero(np.isnan(fltAreaChangeAll))[0]
#plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
#plt.colorbar()

axGLflux = fig.add_subplot(nrow, ncol, 4)
plt.title(f'Grounding line flux at year {targetYear} (Gt)')
plt.xlabel('von Mises stress threshold (kPa)')
plt.ylabel('basal friction law exponent')
plt.grid()
plt.scatter(vmThresh, fricExp, s=markerSize, c=GLfluxAll, plotnonfinite=False)
badIdx = np.nonzero(np.isnan(GLfluxAll))[0]
plt.plot(vmThresh[badIdx], fricExp[badIdx], 'kx')
plt.colorbar()


fig.tight_layout()
plt.show()
