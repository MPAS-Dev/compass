import xarray
import os
import numpy

from mpas_tools.io import write_netcdf

from compass.step import Step
from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter, \
    TimeSeriesPlotter
from compass.ocean.haney import compute_haney_number


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
    def __init__(self, test_case, resolution, experiment):
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

        # show progress only if we're not writing to a log file
        show_progress = self.log_filename is None

        sim_dir = '../simulation'
        streamfunction_dir = '../streamfunction'
        out_dir = '.'
        expt = self.experiment

        dsMesh = xarray.open_dataset('{}/init.nc'.format(sim_dir))
        dsOut = xarray.open_dataset('{}/output.nc'.format(sim_dir))
        dsForcing = xarray.open_dataset('{}/forcing_data_init.nc'.format(sim_dir))

        plotter = MoviePlotter(inFolder=sim_dir,
                                streamfunctionFolder=streamfunction_dir,
                                outFolder='{}/plots'.format(out_dir),
                                expt=expt, sectionY=section_y,
                                dsMesh=dsMesh, ds=dsOut,
                                showProgress=show_progress)


        delssh = dsOut.ssh-dsOut.ssh[0,:]
        plotter.plot_horiz_series(dsOut.ssh, 'ssh', 'ssh', True, cmap='cmo.curl')
        plotter.plot_horiz_series(dsOut.velocityX[:,:,0], 'u', 'u', True,
                                  cmap='cmo.balance')
        plotter.plot_horiz_series(dsOut.velocityY[:,:,0], 'v', 'v', True,
                                  cmap='cmo.balance')
        plotter.plot_horiz_series(dsOut.ssh + dsMesh.bottomDepth, 'H', 'H', True,
                                  vmin=3e-3+1e-10, vmax=10, cmap_set_under='r')
        plotter.plot_horiz_series(delssh, 'delssh', 'delssh', True,
                                  cmap='cmo.curl', vmin=-1, vmax=1)
        delice = dsOut.landIcePressure-dsOut.landIcePressure[0,:]
        plotter.plot_horiz_series(delice, 'delLandIcePressure', 'delLandIcePressure', 
                                  True, cmap='cmo.curl')

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
