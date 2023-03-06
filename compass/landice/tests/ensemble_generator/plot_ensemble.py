#!/usr/bin/env python

import configparser
import glob
import os

import matplotlib.tri as tri
import netCDF4
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

# --------------
# general settings
# --------------
targetYear = 20.0  # model year from start at which to calculate statistics
labelRuns = True
lw = 0.5  # linewidth for ensemble plots

# physical constants
rhoi = 910.0
rhosw = 1028.0

# find list of runs
runs = sorted(glob.glob("run*"))
nRuns = len(runs)

# --------------
# Set up data structures
# --------------

# Set up nested dictionary for possible parameters
# These are the parameters supported by the script.
# The script will determine which ones are active in a given ensemble.
# The values array is 1d array of values from each run
param_info = {
    'basal_fric_exp': {'units': 'unitless',
                       'values': np.zeros((nRuns,)) * np.nan},
    'von_mises_threshold': {'units': 'Pa',
                            'values': np.zeros((nRuns,)) * np.nan},
    'calv_spd_limit': {'units': 'm/s',
                       'values': np.zeros((nRuns,)) * np.nan},
    'mu_scale': {'units': 'unitless',
                 'values': np.zeros((nRuns,)) * np.nan},
    'stiff_scale': {'units': 'unitless',
                    'values': np.zeros((nRuns,)) * np.nan},
    'gamma0': {'units': 'unitless',
               'values': np.zeros((nRuns,)) * np.nan},
    'deltaT': {'units': 'deg C',
               'values': np.zeros((nRuns,)) * np.nan}}

# Set up nested dictionary for possible quantities of interest.
# The values array is 1d array of values from each run
qoi_info = {
    'SLR': {'title': f'SLR at year {targetYear}',
            'units': 'mm',
            'values': np.zeros((nRuns,)) * np.nan},
    'total area': {'title': f'Total area change at year {targetYear}',
                   'units': 'km$^2$',
                   'values': np.zeros((nRuns,)) * np.nan},
    'grd area': {'title': f'Grounded area change at year {targetYear}',
                 'units': 'km$^2$',
                 'values': np.zeros((nRuns,)) * np.nan},
    'GL flux': {'title': f'Grounding line flux at year {targetYear}',
                'units': 'Gt/yr',
                'values': np.zeros((nRuns,)) * np.nan}}

# --------------
# Observations information - to eventually moved to a compass module
# --------------

# obs from Rignot et al 2019 PNAS
obs_discharge_thwaites = np.array([
    82.42, 83.25, 84.08, 84.90, 85.73, 86.56, 87.39, 88.22, 89.04, 89.87,
    90.70, 91.40, 92.10, 92.80, 93.50, 94.20, 94.90, 95.60, 95.90, 96.20,
    96.50, 96.80, 97.10, 97.40, 97.75, 98.10, 101.60, 101.40, 102.80,
    104.30, 109.70, 114.10, 115.80, 118.70, 121.60, 123.80, 117.20, 115.10,
    119.68])
obs_discharge_haynes = np.array([
    10.10, 10.25, 10.40, 10.55, 10.70, 10.85, 11.00, 11.15, 11.30, 11.45,
    11.60, 11.49, 11.37, 11.26, 11.14, 11.03, 10.91, 10.80, 10.93, 11.07,
    11.20, 11.33, 11.47, 11.60, 11.35, 11.10, 12.20, 11.90, 12.50, 12.11,
    12.80, 12.80, 12.90, 12.80, 13.90, 14.70, 12.30, 11.60, 12.53])
obs_discharge = obs_discharge_thwaites + obs_discharge_haynes
obs_years = np.arange(1979, 2017 + 1) - 2000.0
obs_sigmaD = (3.93**2 + 1.00**2)**0.5  # thwaites + haynes

# obs from Adusumilli et al 2020 Supp Table 1
obs_melt_yrs = [10, 18]
obs_melt = 81.1
obs_melt_unc = 7.4
obs_melt_yrs = np.array([2003, 2008]) - 2000
obs_melt = 97.5
obs_melt_unc = 7.0

# --------------
# Set up time series plots
# --------------

# Set up axes for time series plots before reading data.
# Time series are plotted as they are read.
figTS = plt.figure(1, figsize=(8, 12), facecolor='w')
nrow = 6
ncol = 1
axSLRts = figTS.add_subplot(nrow, ncol, 1)
plt.xlabel('Year')
plt.ylabel('SLR\ncontribution\n(mm)')
plt.grid()

axTAts = figTS.add_subplot(nrow, ncol, 2, sharex=axSLRts)
plt.xlabel('Year')
plt.ylabel('Total area\nchange (km2)')
plt.grid()

axGAts = figTS.add_subplot(nrow, ncol, 3, sharex=axSLRts)
plt.xlabel('Year')
plt.ylabel('Grounded area\nchange (km2)')
plt.grid()

axFAts = figTS.add_subplot(nrow, ncol, 4, sharex=axSLRts)
plt.xlabel('Year')
plt.ylabel('Floating\narea (km2)')
plt.grid()

axBMBts = figTS.add_subplot(nrow, ncol, 5, sharex=axSLRts)
plt.xlabel('Year')
plt.ylabel('Ice-shelf\nbasal melt\nflux (Gt/yr)')
plt.grid()
axBMBts.fill_between(obs_melt_yrs, obs_melt - obs_melt_unc, obs_melt +
                     obs_melt_unc, color='k', alpha=0.8, label='melt obs')

axGLFts = figTS.add_subplot(nrow, ncol, 6, sharex=axSLRts)
plt.xlabel('Year')
plt.ylabel('GL flux\n(Gt/yr)')
plt.grid()
axGLFts.fill_between(obs_years, obs_discharge - 2.0 * obs_sigmaD,
                     obs_discharge + 2.0 * obs_sigmaD, color='k', alpha=0.8,
                     label='D obs')

# --------------
# maps plotting setup
# --------------

figMaps = plt.figure(2, figsize=(8, 12), facecolor='w')
nrow = 2
ncol = 1
axMaps = figMaps.add_subplot(nrow, ncol, 1)
axMaps.axis('equal')
axMaps2 = figMaps.add_subplot(nrow, ncol, 2)
axMaps2.axis('equal')

firstMap = True

GLX = np.array([])
GLY = np.array([])

# --------------
# Loop through runs and gather data
# --------------
for idx, run in enumerate(runs):
    print(f'Analyzing {run}')
    # get param values for this run
    run_cfg = configparser.ConfigParser()
    run_cfg.read(os.path.join(run, 'run_info.cfg'))
    run_info = run_cfg['run_info']

    # Loop through params and find their values if active
    for param in param_info:
        if param in run_info:
            param_info[param]['active'] = True
            param_info[param]['values'][idx] = run_info.get(param)
        else:
            param_info[param]['active'] = False

    fpath = run + "/output/globalStats.nc"
    if os.path.exists(fpath):
        f = netCDF4.Dataset(fpath, 'r')
        years = f.variables['daysSinceStart'][:] / 365.0

        VAF = f.variables['volumeAboveFloatation'][:]
        SLR = (VAF[0] - VAF) / 3.62e14 * rhoi / rhosw * 1000.
        totalArea = f.variables['totalIceArea'][:] / 1.0e6  # in km2
        fltArea = f.variables['floatingIceArea'][:] / 1.0e6  # in km2
        grdArea = f.variables['groundedIceArea'][:] / 1.0e6  # in km2
        groundingLineFlux = f.variables['groundingLineFlux'][:] / 1.0e12  # Gt
        BMB = f.variables['totalFloatingBasalMassBal'][:] / -1.0e12  # in Gt

        # plot time series
        axSLRts.plot(years, SLR, linewidth=lw)
        axTAts.plot(years, totalArea - totalArea[0], linewidth=lw)
        axGAts.plot(years, grdArea - grdArea[0], linewidth=lw)
        axFAts.plot(years, fltArea, linewidth=lw)
        # ignore first entry which is 0
        axGLFts.plot(years[1:], groundingLineFlux[1:], linewidth=lw)
        # ignore first entry which is 0
        axBMBts.plot(years[1:], BMB[1:], linewidth=lw)

        # Only process runs that have reached target year
        indices = np.nonzero(years >= targetYear)[0]
        if len(indices) > 0:
            ii = indices[0]
            print(f'{run} using year {years[ii]}')
            qoi_info['SLR']['values'][idx] = SLR[ii]

            grdArea = f.variables['groundedIceArea'][:] / 1000.0**2  # in km^2
            qoi_info['grd area']['values'][idx] = grdArea[ii] - grdArea[0]

            iceArea = f.variables['totalIceArea'][:] / 1000.0**2  # in km^2
            qoi_info['total area']['values'][idx] = iceArea[ii] - iceArea[0]

            qoi_info['GL flux']['values'][idx] = groundingLineFlux[ii]

        # plot map
        DS = xr.open_mfdataset(run + '/output/' + 'output_*.nc',
                               combine='nested', concat_dim='Time',
                               decode_timedelta=False,
                               chunks={"Time": 10})
        yearsOutput = DS['daysSinceStart'].values[:] / 365.0
        indices = np.nonzero(yearsOutput >= targetYear)[0]
        if len(indices) > 0:
            ii = indices[0]

            thickness = DS['thickness'].values
            bedTopo = DS['bedTopography'].values
            xCell = DS['xCell'].values[0, :]
            yCell = DS['yCell'].values[0, :]

            triang = tri.Triangulation(xCell, yCell)
            grd = ((thickness[ii, :] * 910.0 / 1028.0 + bedTopo[ii, :]) >
                   0.0) * (thickness[ii, :] > 0.0)

            if firstMap is True:
                firstMap = False
                axMaps.tricontour(triang, thickness[0], [1.0], colors='k',
                                  linewidths=3)
                grd0 = ((thickness[0, :] * 910.0 / 1028.0 + bedTopo[0, :]) >
                        0.0) * (thickness[0, :] > 0.0)
                axMaps.tricontour(triang, grd0, [0.5], colors='k',
                                  linewidths=3)

            axMaps.tricontour(triang, thickness[ii], [1.0], colors='r',
                              linewidths=lw)
            grdcontourset = axMaps.tricontour(triang, grd, [0.5], colors='b',
                                              linewidths=lw)

            GLX = np.append(GLX, grdcontourset.allsegs[0][0][:, 0])
            GLY = np.append(GLY, grdcontourset.allsegs[0][0][:, 1])

        f.close()

# --------------
# finalize plots generated during data reading
# --------------

# axMaps2.plot(GLX, GLY, '.')
axMaps2.hist2d(GLX, GLY, (50, 50), cmap=plt.cm.jet)
figMaps.savefig('figure_maps.png')

figTS.tight_layout()
figTS.savefig('figure_time_series.png')

# --------------
# single parameter plots
# --------------

fig_num = 0
fig_offset = 100
for param in param_info:
    if param_info[param]['active']:
        fig = plt.figure(fig_offset + fig_num, figsize=(10, 8), facecolor='w')
        nrow = 2
        ncol = 2
        fig.suptitle(f'{param} sensitivities')
        # create subplot for each QOI
        n_sub = 1
        for qoi in qoi_info:
            ax = fig.add_subplot(nrow, ncol, n_sub)
            plt.title(qoi_info[qoi]['title'])
            plt.xlabel(f'{param} ({param_info[param]["units"]})')
            plt.ylabel(f'{qoi} ({qoi_info[qoi]["units"]})')
            plt.plot(param_info[param]['values'], qoi_info[qoi]['values'],
                     '.')
            if labelRuns:
                for i in range(nRuns):
                    plt.annotate(f'{runs[i][3:]}',
                                 (param_info[param]['values'][i],
                                  qoi_info[qoi]['values'][i]))
            n_sub += 1
        fig.tight_layout()
        fig.savefig(f'figure_sensitivity_{param}.png')
        fig_num += 1

# --------------
# pairwise parameter plots
# --------------

fig_num = 0
fig_offset = 200
markerSize = 100
for count1, param1 in enumerate(param_info):
    if param_info[param1]['active']:
        p1_cnt = count1
        for count2, param2 in enumerate(param_info):
            if count2 > count1 and param_info[param2]['active']:
                fig = plt.figure(fig_offset + fig_num, figsize=(10, 8),
                                 facecolor='w')
                nrow = 2
                ncol = 2
                fig.suptitle(f'{param1} vs. {param2} sensitivities')
                # create subplot for each QOI
                n_sub = 1
                for qoi in qoi_info:
                    ax = fig.add_subplot(nrow, ncol, n_sub)
                    plt.title(f'{qoi_info[qoi]["title"]} '
                              f'({qoi_info[qoi]["units"]})')
                    plt.xlabel(f'{param1} ({param_info[param1]["units"]})')
                    plt.ylabel(f'{param2} ({param_info[param2]["units"]})')
                    xdata = param_info[param1]['values']
                    ydata = param_info[param2]['values']
                    zdata = qoi_info[qoi]['values']
                    plt.scatter(xdata, ydata, s=markerSize, c=zdata,
                                plotnonfinite=False)
                    badIdx = np.nonzero(np.isnan(zdata))[0]
                    plt.plot(xdata[badIdx], ydata[badIdx], 'kx')
                    if labelRuns:
                        for i in range(nRuns):
                            plt.annotate(f'{runs[i][3:]}',
                                         (xdata[i], ydata[i]))
                    plt.colorbar()
                    n_sub += 1
                fig.tight_layout()
                fig.savefig(
                    f'figure_pairwise_sensitivity_{param1}_{param2}.png')
                fig_num += 1

plt.show()
