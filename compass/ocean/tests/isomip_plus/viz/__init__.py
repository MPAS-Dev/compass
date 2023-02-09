import os

import matplotlib.pyplot as plt
import numpy as np
import xarray
from mpas_tools.io import write_netcdf

from compass.ocean.haney import compute_haney_number
from compass.ocean.tests.isomip_plus.viz.plot import (
    MoviePlotter,
    TimeSeriesPlotter,
)
from compass.step import Step


class Viz(Step):
    """
    A step for visualizing output from an ISOMIP+ simulation

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment
    """
    def __init__(self, test_case, resolution, experiment, tidal_forcing=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment
        """
        super().__init__(test_case=test_case, name='viz')
        self.resolution = resolution
        self.experiment = experiment
        self.tidal_forcing = tidal_forcing

    def run(self):
        """
        Run this step of the test case
        """

        config = self.config
        section = config['isomip_plus_viz']
        plot_streamfunctions = section.getboolean('plot_streamfunctions')
        plot_haney = section.getboolean('plot_haney')
        frames_per_second = section.getint('frames_per_second')
        movie_format = section.get('movie_format')
        section_y = section.getfloat('section_y')

        section = config['isomip_plus']
        min_column_thickness = section.getfloat('min_column_thickness')

        # show progress only if we're not writing to a log file
        show_progress = self.log_filename is None

        expt = self.experiment
        if self.tidal_forcing:
            sim_dir = '../performance'
        else:
            sim_dir = '../simulation'
        if not os.path.exists(f'{sim_dir}/timeSeriesStatsMonthly.0001-01-01.nc'):  # noqa: E501
            sim_dir = '../performance'
        streamfunction_dir = '../streamfunction'
        out_dir = '.'

        dsMesh = xarray.open_dataset(f'{sim_dir}/init.nc')
        dsOut = xarray.open_dataset(f'{sim_dir}/output.nc')

        plotter = MoviePlotter(inFolder=sim_dir,
                               streamfunctionFolder=streamfunction_dir,
                               outFolder=f'{out_dir}/plots',
                               expt=expt, sectionY=section_y,
                               dsMesh=dsMesh, ds=dsOut,
                               showProgress=show_progress)

        if 'time_varying' in expt:
            plotter.plot_horiz_series(dsOut.ssh, 'ssh', 'ssh', True,
                                      cmap='cmo.curl')
            delice = dsOut.landIcePressure - dsOut.landIcePressure[0, :]
            plotter.plot_horiz_series(delice, 'delLandIcePressure',
                                      'delLandIcePressure', True,
                                      cmap='cmo.curl')
        plotter.plot_horiz_series(dsOut.velocityX[:, :, 0], 'u', 'u', True,
                                  cmap='cmo.balance', vmin=-5e-1, vmax=5e-1)
        plotter.plot_horiz_series(dsOut.velocityY[:, :, 0], 'v', 'v', True,
                                  cmap='cmo.balance', vmin=-5e-1, vmax=5e-1)
        plotter.plot_horiz_series(dsOut.ssh + dsMesh.bottomDepth, 'H', 'H',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=700, cmap_set_under='r',
                                  cmap_scale='log')

        if 'tidal' in expt:
            delssh = dsOut.ssh - dsOut.ssh[0, :]
            plotter.plot_horiz_series(delssh, 'delssh', 'delssh', True,
                                      cmap='cmo.curl', vmin=-1, vmax=1)

        wct = dsOut.ssh + dsMesh.bottomDepth
        idx_thin = np.logical_and(wct[0, :] > 1e-1,
                                  wct[0, :] < 1)
        wct_thin = wct[:, idx_thin]
        wct_mean = wct_thin.mean(dim='nCells').values
        time = dsOut.daysSinceStartOfSim.values
        fig = plt.figure()
        plt.plot(time, wct_mean, '.')
        fig.set_xlabel('Time (days)')
        fig.set_ylabel('Mean thickness of thin film (m)')
        plt.savefig('wct_thin_t.png')
        plt.close()

        plotter.plot_horiz_series(wct, 'H', 'H',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=700, cmap_set_under='r',
                                  cmap_scale='log')

        if os.path.exists(f'{sim_dir}/timeSeriesStatsMonthly.0001-01-01.nc'):
            ds = xarray.open_mfdataset(
                '{}/timeSeriesStatsMonthly*.nc'.format(sim_dir),
                concat_dim='Time', combine='nested')

            if plot_haney:
                _compute_and_write_haney_number(dsMesh, ds, out_dir,
                                                showProgress=show_progress)

            tsPlotter = TimeSeriesPlotter(inFolder=sim_dir,
                                          outFolder='{}/plots'.format(out_dir),
                                          expt=expt)
            tsPlotter.plot_melt_time_series()
            tsPlotter = TimeSeriesPlotter(
                inFolder=sim_dir,
                outFolder='{}/timeSeriesBelow300m'.format(out_dir),
                expt=expt)
            tsPlotter.plot_melt_time_series(sshMax=-300.)

            mPlotter = MoviePlotter(inFolder=sim_dir,
                                    streamfunctionFolder=streamfunction_dir,
                                    outFolder='{}/plots'.format(out_dir),
                                    expt=expt, sectionY=section_y,
                                    dsMesh=dsMesh, ds=ds,
                                    showProgress=show_progress)

            mPlotter.plot_layer_interfaces()

            if plot_streamfunctions:
                mPlotter.plot_barotropic_streamfunction()
                mPlotter.plot_overturning_streamfunction()

            if plot_haney:
                mPlotter.plot_haney_number(haneyFolder=out_dir)

            mPlotter.plot_melt_rates()
            mPlotter.plot_ice_shelf_boundary_variables()
            mPlotter.plot_temperature()
            mPlotter.plot_salinity()
            mPlotter.plot_potential_density()

            mPlotter.images_to_movies(outFolder='{}/movies'.format(out_dir),
                                      framesPerSecond=frames_per_second,
                                      extension=movie_format)


def file_complete(ds, fileName):
    """
    Find out if the file already has the same number of time slices as the
    monthly-mean data set
    """
    complete = False
    if os.path.exists(fileName):
        with xarray.open_dataset(fileName) as dsCompare:
            if ds.sizes['Time'] == dsCompare.sizes['Time']:
                complete = True

    return complete


def _compute_and_write_haney_number(dsMesh, ds, folder, showProgress=False):
    """
    compute the Haney number rx1 for each edge, and interpolate it to cells.
    """

    haneyFileName = '{}/haney.nc'.format(folder)
    if file_complete(ds, haneyFileName):
        return

    haneyEdge, haneyCell = compute_haney_number(
        dsMesh, ds.timeMonthly_avg_layerThickness, ds.timeMonthly_avg_ssh,
        showProgress)
    dsHaney = xarray.Dataset()
    dsHaney['xtime_startMonthly'] = ds.xtime_startMonthly
    dsHaney['xtime_endMonthly'] = ds.xtime_endMonthly
    dsHaney['haneyEdge'] = haneyEdge
    dsHaney.haneyEdge.attrs['units'] = 'unitless'
    dsHaney.haneyEdge.attrs['description'] = 'Haney number on edges'
    dsHaney['haneyCell'] = haneyCell
    dsHaney.haneyCell.attrs['units'] = 'unitless'
    dsHaney.haneyCell.attrs['description'] = 'Haney number on cells'
    dsHaney = dsHaney.transpose('Time', 'nCells', 'nEdges', 'nVertLevels')
    write_netcdf(dsHaney, haneyFileName)
