#!/usr/bin/env python
'''
Analyze subglacial hydrology time-series from a landice globalStats file
and determine if the simulation has reached steady state.

Steady state is defined as when the water mass balance equation is
approximately satisfied over a 10-year rolling average:

melt + chnlMelt ≈ distFluxMarine + chnlFluxMarine + distFluxLand + chnlFluxLand
'''

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import argparse
import json
import sys

import netCDF4 as nc
import numpy as np
from matplotlib import pyplot as plt

rhow = 1000.0
secyr = 3600.0 * 24.0 * 365.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-f",
    dest="filename",
    help="input filename",
    default="globalStats.nc",
    metavar="FILENAME")
parser.add_argument(
    "-u",
    dest="units",
    help="units for mass: kg, Gt",
    default="Gt",
    metavar="UNITS")
parser.add_argument(
    "--window_years",
    dest="window_years",
    type=float,
    default=10.0,
    help="Rolling window size in years for steady-state check (default: 10.0)")
parser.add_argument(
    "--imbalance_threshold",
    dest="imbalance_threshold",
    type=float,
    default=0.05,
    help=["Relative imbalance threshold for \
        steady state (default: 0.05 = 5% relative error)"])
parser.add_argument(
    "--output_json",
    dest="output_json",
    help="JSON file to save steady-state analysis results",
    default="steadystate_results.json")
parser.add_argument("--plot", dest="plot", action='store_true', default=False,
                    help="Generate plots (default: False)")
args = parser.parse_args()

# Scaling assuming variables are in kg
if args.units == "kg":
    massUnit = "kg"
    fluxUnit = "kg yr$^{-1}$"
    unitScaling = 1.0
elif args.units == "Gt":
    massUnit = "Gt"
    fluxUnit = "Gt yr$^{-1}$"
    unitScaling = 1.0e-12
else:
    sys.exit("Unknown mass unit")

print("Using mass units of: ", massUnit)

dataset = nc.Dataset(args.filename)

# Read variables
# convert everything to kg and years before unit conversion
totalSubglacialWaterMass = \
    dataset.variables['totalSubglacialWaterVolume'][:] * rhow * unitScaling
melt = dataset.variables['totalBasalMeltInput'][:] * unitScaling * secyr
distFluxMarine = dataset.variables['totalDistWaterFluxMarineMargin'][:] * \
    unitScaling * secyr
chnlFluxMarine = dataset.variables['totalChnlWaterFluxMarineMargin'][:] * \
    unitScaling * secyr
distFluxLand = dataset.variables['totalDistWaterFluxTerrestrialMargin'][:] * \
    unitScaling * secyr
chnlFluxLand = dataset.variables['totalChnlWaterFluxTerrestrialMargin'][:] * \
    unitScaling * secyr
chnlMelt = dataset.variables['totalChannelMelt'][:] * unitScaling * secyr
flotFrac = dataset.variables['avgFlotationFraction'][:]
lakeArea = dataset.variables['totalSubglacialLakeArea'][:] / 1000.0**2  # km^2
lakeMass = dataset.variables['totalSubglacialLakeVolume'][:] * \
    rhow * unitScaling
grdArea = dataset.variables['groundedIceArea'][:] / 1000.0**2  # km^2

deltat = dataset.variables['deltat'][:] / secyr  # in years
yr = dataset.variables['daysSinceStart'][:] / 365.0

subglacialWaterMassRate = np.zeros((len(melt),))

for i in range(len(totalSubglacialWaterMass) - 1):
    subglacialWaterMassRate[i] = ((totalSubglacialWaterMass[i + 1] -
                                   totalSubglacialWaterMass[i]) / deltat[i])

# ============================================================================
# STEADY-STATE ANALYSIS
# ============================================================================


def calculate_rolling_average(data, yr_array, window_years):
    """
    Calculate rolling average over a specified time window.

    Parameters
    ----------
    data : array
        Data to average
    yr_array : array
        Time array in years
    window_years : float
        Window size in years

    Returns
    -------
    rolling_avg : array
        Rolling average values
    yr_windows : array
        Time values at center of each window
    """
    rolling_avg = np.full_like(data, np.nan)
    yr_windows = np.full_like(data, np.nan)

    for i in range(len(data)):
        # Find points within window_years of current point
        mask = np.abs(yr_array - yr_array[i]) <= window_years / 2.0
        if np.sum(mask) > 0:
            rolling_avg[i] = np.mean(data[mask])
            yr_windows[i] = yr_array[i]

    return rolling_avg, yr_windows


def check_steady_state(
        yr,
        melt_in,
        chnl_melt_in,
        dist_flux_marine_out,
        chnl_flux_marine_out,
        dist_flux_land_out,
        chnl_flux_land_out,
        window_years=10.0,
        imbalance_threshold=0.05):
    """
    Check if simulation has reached steady state based on water mass balance.

    Steady state is defined as when the mass balance equation is approximately
    satisfied over a rolling time window:

        Input (melt + chnlMelt) ≈ Output (sum of outfluxes)

    Parameters
    ----------
    yr : array
        Time array in years
    melt_in : array
        Basal melt flux
    chnl_melt_in : array
        Channel melt flux
    dist_flux_marine_out : array
        Distributed water flux at marine margin
    chnl_flux_marine_out : array
        Channel water flux at marine margin
    dist_flux_land_out : array
        Distributed water flux at terrestrial margin
    chnl_flux_land_out : array
        Channel water flux at terrestrial margin
    window_years : float
        Rolling window size in years
    imbalance_threshold : float
        Relative imbalance threshold (e.g., 0.05 = 5%)

    Returns
    -------
    is_steady : bool
        Whether simulation appears to be at steady state
    steady_state_metrics : dict
        Dictionary containing steady-state metrics and analysis
    analysis_data : dict
        Dictionary with time series data for plotting
    """

    # Calculate totals
    total_input = melt_in + chnl_melt_in
    total_output = (dist_flux_marine_out + chnl_flux_marine_out +
                    dist_flux_land_out + chnl_flux_land_out)

    # Calculate rolling averages
    input_rolling, _ = calculate_rolling_average(total_input, yr, window_years)
    output_rolling, yr_rolling = calculate_rolling_average(
        total_output, yr, window_years)

    # Calculate mass balance residual
    residual = total_input - total_output
    residual_rolling, _ = calculate_rolling_average(residual, yr, window_years)

    # Calculate relative imbalance: |input - output| / |input + output|
    denominator = np.abs(input_rolling) + np.abs(output_rolling)
    relative_imbalance = np.full_like(denominator, np.nan)
    valid = denominator > 0
    relative_imbalance[valid] = (np.abs(residual_rolling[valid]) /
                                 denominator[valid])

    # Determine steady state: when relative imbalance is below threshold
    # for the final portion of the simulation
    # No steady-state if run doesn't last 1.5x window length
    if yr[-1] < 1.5 * window_years:
        is_steady = False
        final_imbalance = np.nan
    elif np.sum(np.isfinite(relative_imbalance)) > 0:
        final_portion = \
            relative_imbalance[-max(10, len(relative_imbalance) // 10):]
        is_steady = np.nanmean(final_portion) < imbalance_threshold
        final_imbalance = np.nanmean(final_portion)
    else:
        is_steady = False
        final_imbalance = np.nan

    # Find when steady state first achieved (if at all)
    steady_state_idx = None
    if np.sum(relative_imbalance < imbalance_threshold) > 0:
        steady_state_idx = np.where(
            relative_imbalance < imbalance_threshold)[0][0]
        time_to_steady = yr[steady_state_idx]
    else:
        time_to_steady = np.nan

    metrics = {
        'is_steady_state': is_steady,
        'window_years': float(window_years),
        'imbalance_threshold': float(imbalance_threshold),
        'final_year': float(yr[-1]),
        'time_to_steady_state_years': float(time_to_steady)
        if not np.isnan(time_to_steady) else None,
        'final_relative_imbalance': float(final_imbalance),
        'final_input_flux': float(input_rolling[-1])
        if np.isfinite(input_rolling[-1]) else None,
        'final_output_flux': float(output_rolling[-1])
        if np.isfinite(output_rolling[-1]) else None,
        'final_residual': float(residual_rolling[-1])
        if np.isfinite(residual_rolling[-1]) else None,
    }

    analysis_data = {
        'yr': yr,
        'input': total_input,
        'output': total_output,
        'residual': residual,
        'input_rolling': input_rolling,
        'output_rolling': output_rolling,
        'residual_rolling': residual_rolling,
        'relative_imbalance': relative_imbalance,
    }

    return is_steady, metrics, analysis_data


# Perform steady-state check
is_steady, steady_metrics, analysis_data = check_steady_state(
    yr, melt, chnlMelt, distFluxMarine, chnlFluxMarine,
    distFluxLand, chnlFluxLand,
    window_years=args.window_years,
    imbalance_threshold=args.imbalance_threshold
)

print("\n" + "=" * 60)
print("STEADY-STATE ANALYSIS")
print("=" * 60)
print(f"Window size: {args.window_years} years")
print(f"Imbalance threshold: {args.imbalance_threshold * 100:.1f}%")
print(f"Final simulation year: {steady_metrics['final_year']:.1f}")
print(
    f"Final relative imbalance: {
        steady_metrics['final_relative_imbalance'] * 100:.2f}%")
if steady_metrics['time_to_steady_state_years'] is not None:
    print(
        f"Time to reach steady state: {
            steady_metrics['time_to_steady_state_years']:.1f} years")
else:
    print("Time to reach steady state: NOT REACHED")
print(f"Is at steady state: {'YES' if is_steady else 'NO'}")
print("=" * 60 + "\n")

# ============================================================================
# PLOTTING
# ============================================================================

if args.plot:
    # Plot 1: Mass balance time-series
    fig, ax = plt.subplots(1, 1, layout='tight', figsize=(10, 6))

    # Input
    plt.plot(yr, melt, 'r:', label='basal melt', linewidth=1.5)
    plt.plot(yr, chnlMelt, 'r--', label='channel melt', linewidth=1.5)
    total_melt = melt + chnlMelt
    plt.plot(yr, total_melt, 'r-', label='total melt (input)', linewidth=2)

    # Output
    plt.plot(
        yr,
        distFluxMarine,
        'b--',
        label='marine sheet outflux',
        linewidth=1.5,
        alpha=0.7)
    plt.plot(
        yr,
        distFluxLand,
        'b:',
        label='land sheet outflux',
        linewidth=1.5,
        alpha=0.7)
    plt.plot(
        yr,
        chnlFluxMarine,
        'c--',
        label='marine chnl outflux',
        linewidth=1.5,
        alpha=0.7)
    plt.plot(
        yr,
        chnlFluxLand,
        'c:',
        label='land chnl outflux',
        linewidth=1.5,
        alpha=0.7)
    total_outflux = (distFluxMarine + distFluxLand +
                     chnlFluxMarine + chnlFluxLand)

    plt.plot(yr, total_outflux, 'b-', lw=2.5, label='total outflux (output)')

    plt.plot(yr[1:-1], subglacialWaterMassRate[1:-1],
             'g-', label='dV/dt', linewidth=2)

    # Plot rolling averages
    plt.plot(
        analysis_data['yr'],
        analysis_data['input_rolling'],
        'r-',
        alpha=0.4,
        linewidth=1,
        label=f'input rolling avg ({
            args.window_years} yr)')
    plt.plot(
        analysis_data['yr'],
        analysis_data['output_rolling'],
        'b-',
        alpha=0.4,
        linewidth=1,
        label=f'output rolling avg ({
            args.window_years} yr)')

    plt.legend(loc='best', fontsize=9, ncol=2)
    plt.xlabel('Year')
    plt.ylabel(f'Mass flux ({fluxUnit})')
    plt.title('Subglacial Water Mass Balance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "subglacial_water_mass_balance.png",
        dpi=300,
        bbox_inches="tight")

    # Plot 2: Mass balance residual
    fig, axes = plt.subplots(
        2, 1, layout='tight', figsize=(
            10, 8), sharex=True)

    # Absolute residual
    axes[0].plot(
        yr,
        analysis_data['residual'],
        'k-',
        linewidth=1.5,
        label='Residual (input - output)')
    axes[0].plot(
        analysis_data['yr'],
        analysis_data['residual_rolling'],
        'r-',
        linewidth=2,
        label=f'Rolling average ({
            args.window_years} yr)')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].fill_between(
        yr,
        0,
        analysis_data['residual'],
        alpha=0.2,
        color='gray')
    axes[0].set_ylabel(f'Residual ({fluxUnit})')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Water Mass Balance Residual')

    # Relative imbalance
    axes[1].plot(
        analysis_data['yr'],
        analysis_data['relative_imbalance'] *
        100,
        'k-',
        linewidth=1.5)
    axes[1].axhline(args.imbalance_threshold * 100,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label='Steady-state threshold \
                    ({args.imbalance_threshold * 100:.1f}%)')
    axes[1].fill_between(
        analysis_data['yr'],
        0,
        analysis_data['relative_imbalance'] *
        100,
        alpha=0.2,
        color='gray')
    axes[1].set_ylabel('Relative Imbalance (%)')
    axes[1].set_xlabel('Year')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Relative Mass Balance Imbalance')

    plt.savefig(
        "water_mass_balance_residual.png",
        dpi=300,
        bbox_inches="tight")

    # Plot 3: Other time-series
    fig, axes = plt.subplots(
        2, 2, sharex=True, layout='tight', figsize=(
            10, 7))
    axes = axes.flatten()

    ax = 0
    axes[ax].plot(yr, flotFrac)
    axes[ax].set_ylabel('Flotation fraction')
    axes[ax].grid(True, alpha=0.3)

    ax += 1
    axes[ax].plot(yr, totalSubglacialWaterMass)
    axes[ax].set_ylabel(f'Water mass ({massUnit})')
    axes[ax].grid(True, alpha=0.3)

    ax += 1
    axes[ax].plot(yr, lakeArea)
    axes[ax].set_ylabel('Lake area (km$^2$)')
    axes[ax].grid(True, alpha=0.3)
    # second axis for % area
    ax2 = axes[ax].twinx()
    ax2.plot(yr, lakeArea / grdArea, ':', color="blue")
    ax2.set_ylabel("Lake area percentage", color="blue")
    ax2.tick_params(axis="y", colors="blue")

    ax += 1
    axes[ax].plot(yr, lakeMass)
    axes[ax].set_ylabel(f'Lake mass ({massUnit})')
    axes[ax].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Year")

    plt.savefig(
        "subglacial_hydrology_timeseries.png",
        dpi=300,
        bbox_inches="tight")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'analysis_type': 'steady_state_water_mass_balance',
    'is_steady_state': is_steady,
    'metrics': steady_metrics,
    'file': args.filename,
}


def convert_to_serializable(obj):
    """Convert numpy/non-serializable types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    else:
        return obj


results = convert_to_serializable(results)

with open(args.output_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {args.output_json}")

dataset.close()
