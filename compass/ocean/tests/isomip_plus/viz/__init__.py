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
        plot_performance_fields = section.getboolean('plot_performance_fields')
        plot_haney = section.getboolean('plot_haney')
        frames_per_second = section.getint('frames_per_second')
        # output_slices = section.getint('output_slices')
        movie_format = section.get('movie_format')
        section_y = section.getfloat('section_y')

        section = config['isomip_plus']
        min_column_thickness = section.getfloat('min_column_thickness')

        # show progress only if we're not writing to a log file
        show_progress = self.log_filename is None

        expt = self.experiment
        sim_dir = '../simulation'
        streamfunction_dir = '../streamfunction'
        out_dir = '.'

        dsMesh = xarray.open_dataset(f'{sim_dir}/init.nc')
        dsOut = xarray.open_dataset(f'{sim_dir}/output.nc')
        dsIce = xarray.open_dataset(f'{sim_dir}/land_ice_fluxes.nc')
        if dsOut.sizes['Time'] < 50:
            print('Cropping dsOut')
            dsOut = dsOut.isel(Time=slice(-31, -1))
        nTime = dsOut.sizes['Time']
        if nTime != dsIce.sizes['Time']:
            print('Cropping dsIce to match dsOut')
            dsIce = dsIce.isel(Time=slice(-nTime - 1, -1))
        dsOut['landIceDraft'] = dsIce.landIceDraft
        plotter = MoviePlotter(inFolder=sim_dir,
                               streamfunctionFolder=streamfunction_dir,
                               outFolder=f'{out_dir}/plots',
                               expt=expt, sectionY=section_y,
                               dsMesh=dsMesh, ds=dsOut,
                               showProgress=show_progress)
        plotter.plot_3d_field_top_bot_section(dsOut.velocityX, 'u', 'u', True,
                                              cmap='cmo.balance',
                                              vmin=-5e-1, vmax=5e-1)
        plotter.plot_3d_field_top_bot_section(dsOut.velocityY, 'v', 'v', True,
                                              cmap='cmo.balance',
                                              vmin=-5e-1, vmax=5e-1)

        # Compute thermal forcing for whole water column
        # landIceInterfaceSalinity and landIceInterfaceTemperature could be
        # used for thermal forcing at the interface
        salinity = dsOut.salinity.isel(nVertLevels=0)
        pressure = dsIce.landIcePressure
        temperature = dsOut.temperature.isel(nVertLevels=0)
        coeff_0 = 6.22e-2
        coeff_S = -5.63e-2
        coeff_p = -7.43e-8
        coeff_pS = -1.74e-10
        coeff_mushy = 0.
        ocn_freezing_temperature = \
            coeff_0 + coeff_S * salinity + \
            coeff_p * pressure + \
            coeff_pS * pressure * salinity + \
            coeff_mushy * salinity / (1.0 - salinity / 1e3)
        dsIce['thermalForcing'] = temperature - ocn_freezing_temperature

        if dsIce.sizes['Time'] > 14:
            dsIce = dsIce.isel(Time=slice(-13, -12))
        if 'xIsomipCell' in dsMesh.keys():
            xCell = dsMesh.xIsomipCell / 1.e3
        else:
            xCell = dsMesh.xCell / 1.e3

        if dsOut.sizes['Time'] > 24:
            days = 30
            samples_per_day = 1
            dsAve = dsOut.isel(Time=slice(-int(days * samples_per_day), -1))
            velocityMagSq = (dsAve.velocityX.isel(nVertLevels=0)**2. +
                             dsAve.velocityX.isel(nVertLevels=0)**2.)
            rmsVelocity = xarray.DataArray(
                data=np.sqrt(velocityMagSq.mean('Time').values),
                dims=['nCells']).expand_dims(dim='Time', axis=0)
            plotter.plot_horiz_series(rmsVelocity,
                                      'rmsVelocity', 'rmsVelocity',
                                      False, vmin=1.e-2, vmax=1.e0,
                                      cmap='cmo.speed', cmap_scale='log')
            cavityMask = np.where(dsMesh.landIceFloatingMask == 1)[0]
            dsCavity = dsIce.isel(nCells=cavityMask)
            rmsVelocityCavity = rmsVelocity.isel(nCells=cavityMask)
            xMasked = xCell.isel(nCells=cavityMask)
            _plot_transect(dsCavity, rmsVelocityCavity, xMasked)

        maxBottomDepth = dsMesh.bottomDepth.max().values
        plotter.plot_horiz_series(dsOut.ssh, 'ssh', 'ssh', True,
                                  vmin=-maxBottomDepth, vmax=0)
        if 'layerThickness' in dsOut.keys():
            plotter.plot_layer_interfaces()
        plotter.plot_horiz_series(dsOut.ssh + dsMesh.bottomDepth, 'H', 'H',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=500, cmap_set_under='k',
                                  cmap_scale='log')
        plotter.plot_horiz_series(dsOut.landIcePressure,
                                  'landIcePressure', 'landIcePressure',
                                  True, vmin=1e5, vmax=1e7, cmap_scale='log')
        plotter.plot_horiz_series(dsOut.surfacePressure,
                                  'surfacePressure', 'surfacePressure',
                                  True, vmin=1e5, vmax=1e7, cmap_scale='log')
        delSurfacePressure = dsOut.landIcePressure - dsOut.surfacePressure
        plotter.plot_horiz_series(delSurfacePressure,
                                  'delSurfacePressure', 'delSurfacePressure',
                                  True, vmin=1e5, vmax=1e7, cmap_scale='log')
        delice = dsOut.landIcePressure - dsMesh.landIcePressure.isel(Time=0)
        plotter.plot_horiz_series(delice, 'delLandIcePressure',
                                  'delLandIcePressure', True,
                                  vmin=-1e6, vmax=1e6, cmap='cmo.balance')
        delssh = dsOut.ssh - dsMesh.ssh.isel(Time=0)
        plotter.plot_horiz_series(delssh, 'ssh - ssh_init', 'delssh_init',
                                  True, cmap='cmo.curl', vmin=-1, vmax=1)
        # Cannot currently plot because it is on edges
        for t in range(dsOut.sizes['Time']):
            fig = plt.figure()
            sc = plt.scatter(
                dsOut.xEdge, dsOut.yEdge, 10,
                dsOut.wettingVelocityFactor.isel(nVertLevels=0, Time=t),
                vmin=0, vmax=1, cmap='cmo.thermal')
            current_cmap = sc.get_cmap()
            current_cmap.set_under('k')
            current_cmap.set_over('r')
            plt.colorbar()
            fig.gca().set_aspect('equal')
            plt.savefig(f'wettingVelocityFactor_{t:02g}.png')
            plt.close(fig)

        wct = dsOut.ssh + dsMesh.bottomDepth
        plotter.plot_horiz_series(wct, 'H', 'H',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=700, cmap_set_under='k',
                                  cmap_scale='log')
        if os.path.exists('../simulation/land_ice_fluxes.nc'):
            ds = xarray.open_mfdataset(
                '../simulation/land_ice_fluxes.nc',
                concat_dim='Time', combine='nested')
            if ds.sizes['Time'] > 50 or \
                    ds.sizes['Time'] != dsOut.sizes['Time']:
                # assumes ds.sizes['Time'] > dsOut.sizes['Time']
                ds = ds.isel(Time=slice(-31, -1))
                dsOut = dsOut.isel(Time=slice(-31, -1))
            if ds.sizes['Time'] != dsOut.sizes['Time']:
                print('error in time selection')
                return
            ds['layerThickness'] = dsOut.layerThickness
            mPlotter = MoviePlotter(inFolder=sim_dir,
                                    streamfunctionFolder=streamfunction_dir,
                                    outFolder='{}/plots'.format(out_dir),
                                    expt=expt, sectionY=section_y,
                                    dsMesh=dsMesh, ds=ds,
                                    showProgress=show_progress)
            _plot_land_ice_variables(ds, mPlotter, maxBottomDepth)

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
        tsPlotter.plot_melt_time_series(wctMax=20.)
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

        mPlotter.images_to_movies(outFolder=f'{out_dir}/movies',
                                  framesPerSecond=frames_per_second,
                                  extension=movie_format)

        if plot_performance_fields:
            plot_folder = f'{out_dir}/performance_plots'
            min_column_thickness = self.config.getfloat('isomip_plus',
                                                        'min_column_thickness')

            ds = xarray.open_dataset('../performance/output.nc')
            # show progress only if we're not writing to a log file
            show_progress = self.log_filename is None
            plotter = MoviePlotter(inFolder=self.work_dir,
                                   streamfunctionFolder=self.work_dir,
                                   outFolder=plot_folder, sectionY=section_y,
                                   dsMesh=dsMesh, ds=ds, expt=self.experiment,
                                   showProgress=show_progress)

            bottomDepth = ds.bottomDepth.expand_dims(dim='Time', axis=0)
            plotter.plot_horiz_series(ds.ssh.isel(Time=-1) + bottomDepth,
                                      'H', 'H', True,
                                      vmin=min_column_thickness, vmax=700,
                                      cmap_set_under='r', cmap_scale='log')
            plotter.plot_horiz_series(ds.ssh, 'ssh', 'ssh',
                                      True, vmin=-700, vmax=0)
            plotter.plot_3d_field_top_bot_section(
                ds.temperature, nameInTitle='temperature', prefix='temp',
                units='C', vmin=-2., vmax=1., cmap='cmo.thermal')

            plotter.plot_3d_field_top_bot_section(
                ds.salinity, nameInTitle='salinity', prefix='salin',
                units='PSU', vmin=33.8, vmax=34.7, cmap='cmo.haline')

            dsIce = xarray.open_dataset('../performance/land_ice_fluxes.nc')
            _plot_land_ice_variables(dsIce, plotter, maxBottomDepth)


def _plot_land_ice_variables(ds, mPlotter, maxBottomDepth):
    tol = 1e-10
    mPlotter.plot_horiz_series(ds.landIceFrictionVelocity,
                               'landIceFrictionVelocity',
                               'frictionVelocity',
                               True)
    mPlotter.plot_horiz_series(ds.landIceFreshwaterFlux,
                               'landIceFreshwaterFlux', 'melt',
                               True)
    mPlotter.plot_horiz_series(ds.landIceFloatingFraction,
                               'landIceFloatingFraction',
                               'landIceFloatingFraction',
                               True, vmin=1e-16, vmax=1,
                               cmap_set_under='k')
    mPlotter.plot_horiz_series(ds.landIceDraft,
                               'landIceDraft', 'landIceDraft',
                               True, vmin=-maxBottomDepth, vmax=0.)
    if 'topDragMagnitude' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.topDragMagnitude,
            'topDragMagnitude', 'topDragMagnitude', True,
            vmin=0 + tol, vmax=np.max(ds.topDragMagnitude.values),
            cmap_set_under='k')
    if 'landIceHeatFlux' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.landIceHeatFlux,
            'landIceHeatFlux', 'landIceHeatFlux', True,
            vmin=np.min(ds.landIceHeatFlux.values),
            vmax=np.max(ds.landIceHeatFlux.values))
    if 'landIceInterfaceTemperature' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.landIceInterfaceTemperature,
            'landIceInterfaceTemperature',
            'landIceInterfaceTemperature',
            True,
            vmin=np.min(ds.landIceInterfaceTemperature.values),
            vmax=np.max(ds.landIceInterfaceTemperature.values))
    if 'landIceFreshwaterFlux' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.landIceFreshwaterFlux,
            'landIceFreshwaterFlux', 'landIceFreshwaterFlux', True,
            vmin=0 + tol, vmax=1e-4,
            cmap_set_under='k', cmap_scale='log')
    if 'landIceFraction' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.landIceFraction,
            'landIceFraction', 'landIceFraction', True,
            vmin=0 + tol, vmax=1 - tol,
            cmap='cmo.balance',
            cmap_set_under='k', cmap_set_over='r')
    if 'landIceFloatingFraction' in ds.keys():
        mPlotter.plot_horiz_series(
            ds.landIceFloatingFraction,
            'landIceFloatingFraction', 'landIceFloatingFraction',
            True, vmin=0 + tol, vmax=1 - tol,
            cmap='cmo.balance', cmap_set_under='k', cmap_set_over='r')
    mPlotter.plot_melt_rates()
    mPlotter.plot_ice_shelf_boundary_variables()


def _plot_transect(dsIce, rmsVelocity, xCell, outFolder='plots'):
    dx = 2.
    xmin = xCell.min()
    xmax = xCell.max()
    xbins = np.arange(xmin, xmax, dx)
    for time_slice in range(dsIce.sizes['Time']):
        title = dsIce.xtime.isel(Time=time_slice).values
        meltTransect = np.zeros_like(xbins)
        landIceDraftTransect = np.zeros_like(xbins)
        frictionVelocityTransect = np.zeros_like(xbins)
        temperatureTransect = np.zeros_like(xbins)
        thermalForcingTransect = np.zeros_like(xbins)
        rmsVelocityTransect = np.zeros_like(xbins)
        for i, xmin in enumerate(xbins):
            binMask = np.where(np.logical_and(xCell >= xmin,
                                              xCell < xmin + dx))[0]
            if (np.sum(binMask) < 1):
                continue
            dsTransect = dsIce.isel(nCells=binMask, Time=time_slice)
            meltTransect[i] = dsTransect.landIceFreshwaterFlux.mean()
            landIceDraftTransect[i] = dsTransect.landIceDraft.mean()
            frictionVelocityTransect[i] = \
                dsTransect.landIceFrictionVelocity.mean()
            temperatureTransect[i] = \
                dsTransect.landIceInterfaceTemperature.mean()
            thermalForcingTransect[i] = dsTransect.thermalForcing.mean()
            rmsVelocityTransect[i] = \
                np.mean(rmsVelocity.isel(Time=time_slice, nCells=binMask))

        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(4, 8))
        color = 'darkgreen'
        x = xbins - xbins[0]
        ax[0].set_title(title)
        ax[0].plot(x, landIceDraftTransect, color=color)
        ax[0].set_ylabel('Land ice draft (m)')
        secPerYear = 365 * 24 * 60 * 60
        density = 1026.
        ax[1].plot(x, meltTransect * secPerYear / density, color=color)
        ax[1].set_ylabel('Melt rate (m/yr)')
        ax[2].plot(x, frictionVelocityTransect, color=color)
        ax[2].set_ylabel('Friction velocity (m/s)')
        ax[3].plot(x, thermalForcingTransect, color=color)
        ax[3].set_ylabel(r'ThermalForcing ($^{\circ}$C)')
        ax[3].set_xlabel('Distance along ice flow (km)')
        plt.tight_layout()
        plt.savefig(f'{outFolder}/meanTransect_tend{time_slice:03g}.png',
                    dpi=600, bbox_inches='tight', transparent=True)
        plt.close(fig)


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
