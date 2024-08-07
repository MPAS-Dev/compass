#!/usr/bin/env python

import configparser
import glob
import os
import pickle
import sys

import matplotlib.tri as tri
import netCDF4
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from compass.landice.ais_observations import ais_basin_info

# --------------
# general settings
# --------------
target_year = 50.0  # model year from start at which to calculate statistics
# model year from start defining start of time interval over which to
# calculate rates of change
start_year_rate = 0.0
label_runs = True
filter_runs = False
plot_time_series = True
plot_single_param_sensitivies = False
plot_pairwise_param_sensitivities = False
plot_qoi_histograms = False
plot_maps = False
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
    'meltflux': {'units': 'Gt/yr',
                 'values': np.zeros((nRuns,)) * np.nan},
    'deltaT': {'units': 'deg C',
               'values': np.zeros((nRuns,)) * np.nan}}

# Set up nested dictionary for possible quantities of interest.
# The values array is 1d array of values from each run
qoi_info = {
    'VAF change': {
        'title': f'VAF change at year {target_year}',
        'units': 'Gt'},
    'VAF change rate': {
        'title': f'VAF change rate between year {start_year_rate} and {target_year}',  # noqa
        'units': 'Gt/yr'},
    'total area change': {
        'title': f'Total area change at year {target_year}',
        'units': 'km$^2$'},
    'total area change rate': {
        'title': f'Total area change rate between year {start_year_rate} and {target_year}',  # noqa
        'units': 'km$^2$/yr'},
    'grd area change': {
        'title': f'Grounded area change at year {target_year}',
        'units': 'km$^2$'},
    'grd area change rate': {
        'title': f'Grounded area change rate between year {start_year_rate} and {target_year}',  # noqa
        'units': 'km$^2$/yr'},
    'grd vol change': {
        'title': f'Grounded vol change at year {target_year}',
        'units': 'Gt'},
    'grd vol change rate': {
        'title': f'Grounded vol change rate between year {start_year_rate} and {target_year}',  # noqa
        'units': 'Gt/yr'},
    'GL flux': {
        'title': f'Grounding line flux at year {target_year}',
        'units': 'Gt/yr'},
    'melt flux': {
        'title': f'Ice-shelf basal melt flux at year {target_year}',
        'units': 'Gt/yr'},
    'speed error': {
        'title': f'Speed error at '
                 f'year {target_year}',
        'units': 'std. devs.'},
    'flt speed error': {
        'title': f'Speed error over '
                 f'floating ice at year {target_year}',
        'units': 'std. devs.'},
    'grd speed error': {
        'title': f'Speed error over '
                 f'grounded ice at year {target_year}',
        'units': 'std. devs.'}}
n_qoi = len(qoi_info)

# Initialize some common attributes to be set later
for qoi in qoi_info:
    qoi_info[qoi]['values'] = np.zeros((nRuns,)) * np.nan
    qoi_info[qoi]['obs'] = None

final_time = np.zeros((nRuns,)) * np.nan
run_nums = np.ones((nRuns,), dtype=int) * -1

# Get ensemble-wide information
basin = None
ens_cfg = configparser.ConfigParser()
# Check for presence of two possible cfg file names
ens_cfg_file1 = 'spinup_ensemble.cfg'
ens_cfg_file2 = 'branch_ensemble.cfg'
if os.path.isfile(ens_cfg_file1):
    ens_cfg_file = ens_cfg_file1
elif os.path.isfile(ens_cfg_file2):
    ens_cfg_file = ens_cfg_file2
else:
    sys.exit("A usable cfg file for the ensemble was not found. "
             "Please correct the configuration or disable this check.")
ens_cfg.read(ens_cfg_file)
ens_info = ens_cfg['ensemble']
if 'basin' in ens_info:
    basin = ens_info['basin']
    if basin == 'None':
        basin = None
input_file_path = ens_info['input_file_path']
if basin is None:
    print("No basin found.  Not using observational data.")
else:
    print(f"Using observations from basin {basin} "
          f"({ais_basin_info[basin]['name']}).")


def filter_run():
    valid_run = True
    if filter_runs is True and 'filter criteria' in ais_basin_info[basin]:
        for criterion in ais_basin_info[basin]['filter criteria']:
            # calculate as annual rate
            qoi_val = qoi_info[criterion]['values'][idx]
            filter_info = ais_basin_info[basin]['filter criteria'][criterion]
            min_val = filter_info['values'][0]
            max_val = filter_info['values'][1]
            if qoi_val < min_val or qoi_val > max_val:
                valid_run = False
        # if run marked invalid, set all qoi to nan
        if valid_run is False:
            print(f"Marking run {idx} as invalid due to filtering criteria")
            for qoi in qoi_info:
                qoi_info[qoi]['values'][idx] = np.nan
    return valid_run


# --------------
# Observations information
# --------------

if basin is not None:
    obs_discharge_yrs = np.array([1992., 2006.]) - 2000.0
    obs_discharge, obs_discharge_unc = ais_basin_info[basin]['outflow']

    obs_melt_yrs = np.array([2003., 2008.]) - 2000.0
    obs_melt, obs_melt_unc = ais_basin_info[basin]['shelf_melt']

    qoi_info['GL flux']['obs'] = [obs_discharge, obs_discharge_unc]
    qoi_info['melt flux']['obs'] = [obs_melt, obs_melt_unc]
else:
    obs_discharge_yrs = np.array([0.0, 0.0])
    obs_discharge = np.array([0.0, 0.0])
    obs_discharge_unc = 0.0
    obs_melt_yrs = np.array([0.0, 0.0])
    obs_melt = np.array([0.0, 0.0])
    obs_melt_unc = 0.0

# --------------
# Set up time series plots
# --------------

if plot_time_series:
    # Set up axes for time series plots before reading data.
    # Time series are plotted as they are read.
    fig_ts_mb = plt.figure(1, figsize=(8, 12), facecolor='w')
    nrow = 5
    ncol = 1

    ax_ts_vaf = fig_ts_mb.add_subplot(nrow, ncol, 1)
    plt.ylabel('VAF change\n(Gt)')
    plt.grid()

    ax_ts_grvol = fig_ts_mb.add_subplot(nrow, ncol, 2, sharex=ax_ts_vaf)
    plt.ylabel('Grounded volume\nchange (Gt)')
    plt.grid()

    ax_ts_glf = fig_ts_mb.add_subplot(nrow, ncol, 3, sharex=ax_ts_vaf)
    plt.ylabel('GL flux\n(Gt/yr)')
    plt.grid()
    ax_ts_glf.fill_between(obs_discharge_yrs,
                           obs_discharge - obs_discharge_unc,
                           obs_discharge + obs_discharge_unc,
                           color='b', alpha=0.2, label='D obs')

    ax_ts_glf2 = fig_ts_mb.add_subplot(nrow, ncol, 4, sharex=ax_ts_vaf)
    plt.ylabel('GL flux+\nGL mig. flux\n(Gt/yr)')
    plt.grid()
    ax_ts_glf2.fill_between(obs_discharge_yrs,
                            obs_discharge - obs_discharge_unc,
                            obs_discharge + obs_discharge_unc,
                            color='b', alpha=0.2, label='D obs')

    ax_ts_smb = fig_ts_mb.add_subplot(nrow, ncol, 5, sharex=ax_ts_vaf)
    plt.ylabel('Grounded SMB\n(Gt/yr)')
    plt.xlabel('Year')
    plt.grid()

    # second plot to avoid crowding
    fig_ts_area = plt.figure(2, figsize=(8, 12), facecolor='w')
    nrow = 4
    ncol = 1

    ax_ts_ta = fig_ts_area.add_subplot(nrow, ncol, 1, sharex=ax_ts_vaf)
    plt.ylabel('Total area\nchange (km2)')
    plt.grid()

    ax_ts_ga = fig_ts_area.add_subplot(nrow, ncol, 2, sharex=ax_ts_vaf)
    plt.ylabel('Grounded area\nchange (km2)')
    plt.grid()

    ax_ts_fa = fig_ts_area.add_subplot(nrow, ncol, 3, sharex=ax_ts_vaf)
    plt.ylabel('Floating\narea (km2)')
    plt.grid()

    ax_ts_bmb = fig_ts_area.add_subplot(nrow, ncol, 4, sharex=ax_ts_vaf)
    plt.ylabel('Ice-shelf\nbasal melt\nflux (Gt/yr)')
    plt.xlabel('Year')
    plt.grid()
    ax_ts_bmb.fill_between(obs_melt_yrs,
                           obs_melt - obs_melt_unc,
                           obs_melt + obs_melt_unc,
                           color='b', alpha=0.2, label='melt obs')

# --------------
# maps plotting setup
# --------------

if plot_maps:
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
# Get needed fields from input file
# --------------

DSinput = xr.open_dataset(input_file_path)
observedSurfaceVelocityX = DSinput['observedSurfaceVelocityX'].values[0, :]
observedSurfaceVelocityY = DSinput['observedSurfaceVelocityY'].values[0, :]
obsSpdUnc = DSinput['observedSurfaceVelocityUncertainty'].values[0, :]
obsSpd = (observedSurfaceVelocityX**2 + observedSurfaceVelocityY**2)**0.5
# May want to set speed uncertainty a function of speed, e.g.,
# obsSpdUnc += obsSpd * 0.25
DSinput.close()

# --------------
# Loop through runs and gather data
# --------------
n_dirs = 0
n_with_output = 0
n_at_target_yr = 0
n_filtered = 0
for idx, run in enumerate(runs):
    print(f'Analyzing {run}')
    n_dirs += 1

    run_nums[idx] = int(run[3:])

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
        n_with_output += 1
        f = netCDF4.Dataset(fpath, 'r')
        years = f.variables['daysSinceStart'][:] / 365.0

        final_time[idx] = years[-1]

        VAF = f.variables['volumeAboveFloatation'][:] / \
            (1.0e12 / rhoi)  # in Gt
        totalArea = f.variables['totalIceArea'][:] / 1.0e6  # in km2
        fltArea = f.variables['floatingIceArea'][:] / 1.0e6  # in km2
        grdArea = f.variables['groundedIceArea'][:] / 1.0e6  # in km2
        iceArea = f.variables['totalIceArea'][:] / 1000.0**2  # in km^2
        grdVol = f.variables['groundedIceVolume'][:] / (1.0e12 / rhoi)  # in Gt
        BMB = f.variables['totalFloatingBasalMassBal'][:] / -1.0e12  # in Gt/yr
        grSMB = f.variables['totalGroundedSfcMassBal'][:] / -1.0e12  # in Gt/yr
        groundingLineFlux = f.variables['groundingLineFlux'][:] \
            / 1.0e12  # Gt/yr
        GLMigFlux = f.variables['groundingLineMigrationFlux'][:] \
            / 1.0e12  # Gt/yr
        # Apply smoothing to GL migration flux, because it is very noisy.
        # w is the width of a window over time levels to use for smoothing.
        # May need to play with this to get nice looking GL flux plots.
        # The convolution is a boxcar filter with width w.
        # (Note: if w is larger than the length of your time series, an error
        # will occur.)
        w = 50
        GLMigFlux2 = np.convolve(GLMigFlux, np.ones(w), 'same') / w
        # GLflux2 includes both the groundingLineFlux and the migration flux
        GLflux2 = groundingLineFlux + GLMigFlux2

        # Only process qois for runs that have reached target year
        indices = np.nonzero(years >= target_year)[0]
        if len(indices) > 0:
            n_at_target_yr += 1
            ii = indices[0]
            jj = np.nonzero(years >= start_year_rate)[0][0]
            print(f'{run} using year {years[ii]}. '
                  f'Using start year for rates of {years[jj]}. '
                  f'Max year is {years[-1]}.')

            # find index to start year of rate calculations
            rate_dt = years[ii] - years[jj]

            qoi_info['VAF change']['values'][idx] = VAF[ii] - VAF[0]
            qoi_info['VAF change rate']['values'][idx] = \
                (VAF[ii] - VAF[jj]) / rate_dt
            qoi_info['grd vol change']['values'][idx] = grdVol[ii] - grdVol[0]
            qoi_info['grd vol change rate']['values'][idx] = \
                (grdVol[ii] - grdVol[jj]) / rate_dt
            qoi_info['grd area change']['values'][idx] = (grdArea[ii] -
                                                          grdArea[0])
            qoi_info['grd area change rate']['values'][idx] = \
                (grdArea[ii] - grdArea[jj]) / rate_dt
            qoi_info['total area change']['values'][idx] = (iceArea[ii] -
                                                            iceArea[0])
            qoi_info['total area change rate']['values'][idx] = \
                (iceArea[ii] - iceArea[jj]) / rate_dt
            qoi_info['GL flux']['values'][idx] = groundingLineFlux[ii]
            qoi_info['melt flux']['values'][idx] = BMB[ii]

            # filter run
            valid_run = filter_run()
            if valid_run:
                n_filtered += 1
        else:
            valid_run = False

        # calculate qoi's requiring spatial output
        DS = xr.open_mfdataset(run + '/output/' + 'output_*.nc',
                               combine='nested', concat_dim='Time',
                               decode_timedelta=False,
                               chunks={"Time": 10})
        yearsOutput = DS['daysSinceStart'].values[:] / 365.0
        indices = np.nonzero(yearsOutput >= target_year)[0]
        if len(indices) > 0:
            ii = indices[0]

            surfaceSpeed = DS['surfaceSpeed'].values[ii, :]
            thickness = DS['thickness'].values[ii, :]
            cellMask = DS['cellMask'].values[ii, :]
            areaCell = DS['areaCell'].values[0, :]

            # only evaluate where modeled ice exists and observed speed
            # uncertainy is a reasonable value (<~1e6 m/yr)
            mask = ((thickness > 0.0) *
                    (obsSpdUnc < 0.1) *
                    (obsSpdUnc > 0.0))
            ind = np.nonzero(mask)[0]
            qoi_info['speed error']['values'][idx] = \
                ((areaCell[ind] * (surfaceSpeed[ind] - obsSpd[ind])**2 /
                  obsSpdUnc[ind]**2).sum() / areaCell[ind].sum())**0.5
            # float speed error
            floatMask = (cellMask & 4) // 4
            ind = np.nonzero(mask * floatMask)[0]
            qoi_info['flt speed error']['values'][idx] = \
                ((areaCell[ind] * (surfaceSpeed[ind] - obsSpd[ind])**2 /
                  obsSpdUnc[ind]**2).sum() / areaCell[ind].sum())**0.5
            # grd speed error
            ind = np.nonzero(mask * np.logical_not(floatMask))[0]
            qoi_info['grd speed error']['values'][idx] = \
                ((areaCell[ind] * (surfaceSpeed[ind] - obsSpd[ind])**2 /
                  obsSpdUnc[ind]**2).sum() / areaCell[ind].sum())**0.5
        DS.close()

        if plot_time_series:
            # color lines depending on if they match obs or not
            col = 'k'
            alph = 0.2
            if valid_run:
                col = 'r'
                alph = 0.7

            # plot time series
            # plot 1
            ax_ts_vaf.plot(years, VAF - VAF[0], linewidth=lw, color=col,
                           alpha=alph)
            if label_runs:
                ax_ts_vaf.annotate(run, (years[-1], VAF[-1] - VAF[0]),
                                   fontsize=6)
            ax_ts_grvol.plot(years, grdVol - grdVol[0], linewidth=lw,
                             color=col, alpha=alph)
            if label_runs:
                ax_ts_grvol.annotate(run, (years[-1], grdVol[-1] - grdVol[0]),
                                     fontsize=6)
            # ignore first entry which is 0
            ax_ts_glf.plot(years[1:], groundingLineFlux[1:], linewidth=lw,
                           color=col, alpha=alph)
            # ignore first entry which is 0
            ax_ts_glf2.plot(years[1:], GLflux2[1:], linewidth=lw,
                            color=col, alpha=alph)
            # ignore first entry which is 0
            ax_ts_smb.plot(years[1:], grSMB[1:], linewidth=lw,
                           color=col, alpha=alph)

            # plot 2
            ax_ts_ta.plot(years, totalArea - totalArea[0], linewidth=lw,
                          color=col, alpha=alph)
            if label_runs:
                ax_ts_ta.annotate(run,
                                  (years[-1], totalArea[-1] - totalArea[0]),
                                  fontsize=6)
            ax_ts_ga.plot(years, grdArea - grdArea[0], linewidth=lw,
                          color=col, alpha=alph)
            if label_runs:
                ax_ts_ga.annotate(run, (years[-1], grdArea[-1] - grdArea[0]),
                                  fontsize=6)
            ax_ts_fa.plot(years, fltArea, linewidth=lw, color=col, alpha=alph)
            # ignore first entry which is 0
            ax_ts_bmb.plot(years[1:], BMB[1:], linewidth=lw,
                           color=col, alpha=alph)

        # plot map
        if plot_maps:
            DS = xr.open_mfdataset(run + '/output/' + 'output_*.nc',
                                   combine='nested', concat_dim='Time',
                                   decode_timedelta=False,
                                   chunks={"Time": 10})
            yearsOutput = DS['daysSinceStart'].values[:] / 365.0
            indices = np.nonzero(yearsOutput >= target_year)[0]
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
                    grd0 = ((thickness[0, :] * 910.0 / 1028.0 +
                             bedTopo[0, :]) > 0.0) * (thickness[0, :] > 0.0)
                    axMaps.tricontour(triang, grd0, [0.5], colors='k',
                                      linewidths=3)

                axMaps.tricontour(triang, thickness[ii], [1.0], colors='r',
                                  linewidths=lw)
                grdcontourset = axMaps.tricontour(triang, grd, [0.5],
                                                  colors='b',
                                                  linewidths=lw)

                GLX = np.append(GLX, grdcontourset.allsegs[0][0][:, 0])
                GLY = np.append(GLY, grdcontourset.allsegs[0][0][:, 1])

        f.close()

# Print information about runs
print(f'# runs with directories:     {n_dirs}')
print(f'# runs with output files:    {n_with_output}')
print(f'# runs reaching target year: {n_at_target_yr}')
print(f'# runs passing filter:       {n_filtered}')

# --------------
# save qoi structure
# --------------

data_out = [param_info, qoi_info]
with open('ensemble_data.pickle', 'wb') as f:
    pickle.dump(data_out, f)

# --------------
# finalize plots generated during data reading
# --------------

if plot_maps:
    # axMaps2.plot(GLX, GLY, '.')
    axMaps2.hist2d(GLX, GLY, (50, 50), cmap=plt.get_cmap('jet'))
    figMaps.savefig('figure_maps.png')

if plot_time_series:
    fig_ts_mb.tight_layout()
    fig_ts_mb.savefig('figure_time_series1.png')
    fig_ts_area.tight_layout()
    fig_ts_area.savefig('figure_time_series2.png')

    # optionally add some filtering indicators on the time series
    if filter_runs is True and 'filter criteria' in ais_basin_info[basin]:
        filter_info = ais_basin_info[basin]['filter criteria']

        criterion = 'total area change'
        if criterion in filter_info:
            min_val = filter_info[criterion]['values'][0]
            max_val = filter_info[criterion]['values'][1]
            ax_ts_ta.plot([0, target_year], [min_val, min_val], 'b:')
            ax_ts_ta.plot([0, target_year], [max_val, max_val], 'b:')

        criterion = 'grd area change'
        if criterion in ais_basin_info[basin]['filter criteria']:
            min_val = filter_info[criterion]['values'][0]
            max_val = filter_info[criterion]['values'][1]
            ax_ts_ga.plot([0, target_year], [0, min_val * target_year], 'b:')
            ax_ts_ga.plot([0, target_year], [0, max_val * target_year], 'b:')

        criterion = 'grd vol change'
        if criterion in ais_basin_info[basin]['filter criteria']:
            min_val = filter_info[criterion]['values'][0]
            max_val = filter_info[criterion]['values'][1]
            ax_ts_grvol.plot([0, target_year], [0, min_val * target_year],
                             'b:')
            ax_ts_grvol.plot([0, target_year], [0, max_val * target_year],
                             'b:')

# --------------
# run duration plots
# --------------

fig_duration = plt.figure(99, figsize=(8, 8), facecolor='w')
nrow = 3
ncol = 1

ax_yr_histo = fig_duration.add_subplot(nrow, ncol, 1)
plt.hist(final_time, bins=np.arange(final_time.min(), final_time.max() + 1))
plt.xlabel('final simulated year')
plt.ylabel('count')
plt.grid()

ax_yr_histo_cum = fig_duration.add_subplot(nrow, ncol, 2)
plt.hist(final_time, bins=np.arange(final_time.min(), final_time.max() + 1),
         cumulative=True)
plt.xlabel('final simulated year')
plt.ylabel('cumulative count')
plt.grid()

ax_yr_by_run = fig_duration.add_subplot(nrow, ncol, 3)
plt.bar(run_nums, final_time, width=0.9)
plt.xlabel('run number')
plt.ylabel('final simulated year')
plt.grid(axis='y')

fig_duration.tight_layout()

# --------------
# single parameter plots
# --------------

if plot_single_param_sensitivies:
    fig_num = 0
    fig_offset = 100
    for param in param_info:
        if param_info[param]['active']:
            fig = plt.figure(fig_offset + fig_num, figsize=(13, 8),
                             facecolor='w')
            ncol = int(np.ceil(n_qoi**0.5))
            nrow = int(np.ceil(n_qoi / ncol))
            fig.suptitle(f'{param} sensitivities')
            # create subplot for each QOI
            n_sub = 1
            for qoi in qoi_info:
                ax = fig.add_subplot(nrow, ncol, n_sub)
                plt.title(qoi_info[qoi]['title'])
                plt.xlabel(f'{param} ({param_info[param]["units"]})')
                plt.ylabel(f'{qoi} ({qoi_info[qoi]["units"]})')
                pvalues = param_info[param]['values']
                qvalues = qoi_info[qoi]['values']
                obs = qoi_info[qoi]['obs']
                if obs is not None:
                    plt.fill_between([pvalues.min(), pvalues.max()],
                                     np.array([1., 1.]) * (obs[0] - obs[1]),
                                     np.array([1., 1.]) * (obs[0] + obs[1]),
                                     color='k', alpha=0.2, label='melt obs')
                plt.plot(pvalues, qvalues, '.')
                badIdx = np.nonzero(np.isnan(qvalues))[0]
                plt.plot(pvalues[badIdx], np.zeros(len(badIdx),), 'kx')
                if label_runs:
                    for i in range(nRuns):
                        plt.annotate(f'{runs[i][3:]}',
                                     (pvalues[i], qvalues[i]))
                n_sub += 1
            fig.tight_layout()
            fig.savefig(f'figure_sensitivity_{param}.png')
            fig_num += 1

# --------------
# pairwise parameter plots
# --------------

if plot_pairwise_param_sensitivities:
    fig_num = 0
    fig_offset = 200
    markerSize = 100
    for count1, param1 in enumerate(param_info):
        if param_info[param1]['active']:
            p1_cnt = count1
            for count2, param2 in enumerate(param_info):
                if count2 > count1 and param_info[param2]['active']:
                    fig = plt.figure(fig_offset + fig_num, figsize=(13, 8),
                                     facecolor='w')
                    ncol = int(np.ceil(n_qoi**0.5))
                    nrow = int(np.ceil(n_qoi / ncol))
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
                        if np.isfinite(zdata).sum() == 0:
                            print(f"No valid data for {param1} vs. {param2} "
                                  f"sensitivity plot for {qoi}, skipping")
                            continue
                        plt.scatter(xdata, ydata, s=markerSize, c=zdata,
                                    plotnonfinite=False)
                        badIdx = np.nonzero(np.isnan(zdata))[0]
                        goodIdx = np.nonzero(np.logical_not(np.isnan(
                            zdata)))[0]
                        plt.plot(xdata[badIdx], ydata[badIdx], 'kx')
                        obs = qoi_info[qoi]['obs']
                        plt.colorbar()
                        if obs is not None:
                            try:
                                plt.tricontour(xdata[goodIdx], ydata[goodIdx],
                                               zdata[goodIdx],
                                               [obs[0] - obs[1],
                                                obs[0] + obs[1]],
                                               colors='k')
                            except ValueError:
                                print(f"Skipping obs contour for {param1} "
                                      f"vs. {param2}, because outside model "
                                      "range")
                        if label_runs:
                            for i in range(nRuns):
                                plt.annotate(f'{runs[i][3:]}',
                                             (xdata[i], ydata[i]))
                        n_sub += 1
                    fig.tight_layout()
                    fig.savefig(
                        f'figure_pairwise_sensitivity_{param1}_{param2}.png')
                    fig_num += 1

# --------------
# QOI histograms
# --------------

if plot_qoi_histograms:
    print("Plotting QOI histograms")
    fig = plt.figure(300, figsize=(13, 8), facecolor='w')
    fig.suptitle('QOI histograms')
    ncol = int(np.ceil(n_qoi**0.5))
    nrow = int(np.ceil(n_qoi / ncol))
    n_sub = 0
    for qoi in qoi_info:
        n_sub += 1
        ax = fig.add_subplot(nrow, ncol, n_sub)
        plt.title(qoi_info[qoi]['title'])
        plt.xlabel(f'{qoi} ({qoi_info[qoi]["units"]})')
        plt.ylabel('count')
        qvalues = qoi_info[qoi]['values']
        ax.hist(qvalues)
    fig.tight_layout()
    fig.savefig('figure_qoi_histograms.png')

plt.show()
