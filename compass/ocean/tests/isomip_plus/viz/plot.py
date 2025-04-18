import copy
import glob
import os
import subprocess

import cmocean  # noqa: F401
import matplotlib.pyplot as plt
import numpy
import progressbar
import xarray
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Polygon


class TimeSeriesPlotter(object):
    """
    A plotter object to hold on to some info needed for plotting time series
    from ISOMIP+ simulation results

    Attributes
    ----------
    inFolder : str
        The folder with simulation results

    outFolder : str
        The folder where images will be written

    expt : {'Ocean0', 'Ocean1', 'Ocean2'}
        The name of the experiment
    """
    def __init__(self, inFolder='.', outFolder='plots', expt='Ocean0',
                 dsMesh=None, ds=None):
        """
        Create a plotter object to hold on to some info needed for plotting
        time series from ISOMIP+ simulation results

        Parameters
        ----------
        inFolder : str, optional
            The folder with simulation results

        outFolder : str, optional
            The folder where images will be written

        expt : {'Ocean0', 'Ocean1', 'Ocean2'}, optional
            The name of the experiment

        dsMesh : xarray.Dataset, optional
            The MPAS mesh

        ds : xarray.Dataset, optional
            The time series output
        """

        self.inFolder = inFolder
        self.outFolder = outFolder
        self.expt = expt

        if dsMesh is not None:
            self.dsMesh = dsMesh
        else:
            self.dsMesh = xarray.open_dataset(
                '{}/init.nc'.format(self.inFolder))

        if ds is not None:
            self.ds = ds
        else:
            self.ds = xarray.open_mfdataset(
                '{}/timeSeriesStatsMonthly*.nc'.format(self.inFolder),
                concat_dim='Time', combine='nested')

        try:
            os.makedirs(self.outFolder)
        except OSError:
            pass

        plt.switch_backend('Agg')

    def plot_melt_time_series(self, sshMax=None, wctMax=None):
        """
        Plot a series of image for each of several variables related to melt
        at the ice shelf-ocean interface: mean melt rate, total melt flux,
        mean thermal driving, mean friction velocity
        """

        if 'timeMonthly_avg_landIceFreshwaterFlux' not in self.ds.keys():
            return
        rho_fw = 1000.
        secPerYear = 365 * 24 * 60 * 60

        suffix = ''
        areaCell = self.dsMesh.areaCell
        iceMask = self.ds.timeMonthly_avg_landIceFraction
        meltFlux = self.ds.timeMonthly_avg_landIceFreshwaterFlux
        if sshMax is not None:
            ssh = self.ds.timeMonthly_avg_ssh
            iceMask = iceMask.where(ssh < sshMax)
            suffix = f'_sshMax{sshMax}'
        elif wctMax is not None:
            H = self.ds.timeMonthly_avg_layerThickness.sum(dim='nVertLevels')
            iceMask = iceMask.where(H < wctMax)
            suffix = f'_wctMax{wctMax}'

        totalMeltFlux = (meltFlux * areaCell * iceMask).sum(dim='nCells')
        totalArea = (areaCell * iceMask).sum(dim='nCells')
        meanMeltRate = totalMeltFlux / totalArea / rho_fw * secPerYear
        self.plot_time_series(meanMeltRate, 'mean melt rate', 'meanMeltRate',
                              'm/yr')

        self.plot_time_series(1e-6 * totalMeltFlux, 'total melt flux',
                              f'totalMeltFlux{suffix}', 'kT/yr')

        prefix = 'timeMonthly_avg_landIceBoundaryLayerTracers_'
        boundary_layer_temperature = \
            self.ds[f'{prefix}landIceBoundaryLayerTemperature']
        prefix = 'timeMonthly_avg_landIceInterfaceTracers_'
        interface_temperature = \
            self.ds[f'{prefix}landIceInterfaceTemperature']
        da = boundary_layer_temperature - interface_temperature
        da = (da * areaCell * iceMask).sum(dim='nCells') / totalArea

        self.plot_time_series(da, 'mean thermal driving',
                              f'meanThermalDriving{suffix}', 'deg C')

        da = self.ds.timeMonthly_avg_landIceFrictionVelocity
        da = (da * areaCell * iceMask).sum(dim='nCells') / totalArea

        self.plot_time_series(da, 'mean friction velocity',
                              f'meanFrictionVelocity{suffix}', 'm/s')

    def plot_time_series(self, da, nameInTitle, prefix, units=None,
                         figsize=(12, 6), color=None, overwrite=True):

        fileName = '{}/{}.png'.format(self.outFolder, prefix)
        if not overwrite and os.path.exists(fileName):
            return

        nTime = da.sizes['Time']
        time = numpy.arange(nTime) / 12.

        plt.figure(1, figsize=figsize)
        plt.plot(time, da.values, color=color)

        if units is None:
            ylabel = nameInTitle
        else:
            ylabel = '{} ({})'.format(nameInTitle, units)

        plt.ylabel(ylabel)
        plt.xlabel('time (yrs)')

        plt.savefig(fileName)
        plt.close()


class MoviePlotter(object):
    """
    A plotter object to hold on to some info needed for plotting images from
    ISOMIP+ simulation results

    Attributes
    ----------
    inFolder : str
        The folder with simulation results

    streamfunctionFolder : str
        The folder where streamfunction input files were computed

    outFolder : str
        The folder where images will be written

    expt : {'Ocean0', 'Ocean1', 'Ocean2'}
        The name of the experiment

    sectionY : float
        The location along the y axis of a transect in the x-z plane to plot

    dsMesh : ``xarray.Dataset``
        A data set with mesh data

    ds : ``xarray.Dataset``
        A data set with the montly-mean simulation results

    oceanMask : ``numpy.ndarray``
        A mask of cells that are in the ocean domain (probably all ones)

    cavityMask : ``numpy.ndarray``
        A mask of cells that are in the sub-ice-shelf cavity

    oceanPatches : ``PatchCollection``
        A set of polygons covering ocean cells

    cavityPatches : ``PatchCollection``
        A set of polygons covering only cells in the cavity

    X, Z : ``numpy.ndarray``
        The horiz. and vert. coordinates of the x-z cross section

    sectionMask : ``numpy.ndarray``
        A mask for the cross section indicating where values are valid (i.e.
        above the bathymetry)

    showProgress : bool
        Whether to show a progressbar
    """

    def __init__(self, inFolder, streamfunctionFolder, outFolder, expt,
                 sectionY, dsMesh, ds, showProgress):
        """
        Create a plotter object to hold on to some info needed for plotting
        images from ISOMIP+ simulation results

        Parameters
        ----------
        inFolder : str
            The folder with simulation results

        streamfunctionFolder : str
            The folder where streamfunction input files were computed

        outFolder : str
            The folder where images will be written

        expt : {'Ocean0', 'Ocean1', 'Ocean2'}
            The name of the experiment

        sectionY : float
            The location along the y axis of a transect in the x-z plane to
            plot

        dsMesh : xarray.Dataset
            The MPAS mesh

        ds : xarray.Dataset
            The time series output

        showProgress : bool
            Whether to show a progressbar
        """
        plt.switch_backend('Agg')

        self.inFolder = inFolder
        self.streamfunctionFolder = streamfunctionFolder
        self.outFolder = outFolder
        self.expt = expt
        self.sectionY = sectionY
        self.showProgress = showProgress

        if 'Time' in dsMesh.dims:
            dsMesh = dsMesh.isel(Time=0)
        self.dsMesh = dsMesh
        self.ds = ds

        landIceMask = self.dsMesh.landIceMask > 0
        self.oceanMask = self.dsMesh.maxLevelCell - 1 >= 0
        self.cavityMask = numpy.logical_and(self.oceanMask, landIceMask)

        self.oceanPatches = _compute_cell_patches(
            self.dsMesh, self.oceanMask)
        self.cavityPatches = _compute_cell_patches(
            self.dsMesh, self.cavityMask)

        self.sectionCellIndices = _compute_section_cell_indices(self.sectionY,
                                                                self.dsMesh)

        self._compute_section_x_z()

    def plot_barotropic_streamfunction(self, vmin=None, vmax=None):
        """
        Plot a series of image of the barotropic streamfunction

        Parameters
        ----------
        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar, defaults are
            chosen depending on ``expt``
        """

        ds = xarray.open_dataset('{}/barotropicStreamfunction.nc'.format(
            self.streamfunctionFolder))

        if vmin is None or vmax is None:
            if self.expt in ['Ocean0', 'Ocean1']:
                vmin = -1
                vmax = 1
            else:
                vmin = -0.5
                vmax = 0.5

        nTime = ds.sizes['Time']
        if self.showProgress:
            widgets = ['plotting barotropic streamfunction: ',
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=nTime).start()
        else:
            bar = None

        for tIndex in range(nTime):
            self.update_date(tIndex)
            bsf = ds.bsfCell.isel(Time=tIndex)
            outFileName = '{}/bsf/bsf_{:04d}.png'.format(
                self.outFolder, tIndex + 1)
            self._plot_horiz_field(bsf, title='barotropic streamfunction (Sv)',
                                   outFileName=outFileName, oceanDomain=True,
                                   vmin=vmin, vmax=vmax, cmap='cmo.curl')
            if self.showProgress:
                bar.update(tIndex + 1)
        if self.showProgress:
            bar.finish()

    def plot_overturning_streamfunction(self, vmin=-0.3, vmax=0.3):
        """
        Plot a series of image of the overturning streamfunction

        Parameters
        ----------
        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar
        """

        ds = xarray.open_dataset('{}/overturningStreamfunction.nc'.format(
            self.streamfunctionFolder))

        nTime = ds.sizes['Time']
        if self.showProgress:
            widgets = ['plotting overturning streamfunction: ',
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=nTime).start()
        else:
            bar = None

        for tIndex in range(nTime):
            self.update_date(tIndex)
            osf = ds.osf.isel(Time=tIndex)
            outFileName = '{}/osf/osf_{:04d}.png'.format(self.outFolder,
                                                         tIndex + 1)
            x = _interp_extrap_corner(ds.x.values)
            z = _interp_extrap_corner(ds.z.values)
            self._plot_vert_field(
                x, z, osf, title='overturning streamfunction (Sv)',
                outFileName=outFileName, vmin=vmin, vmax=vmax, cmap='cmo.curl',
                show_boundaries=False)
            if self.showProgress:
                bar.update(tIndex + 1)
        if self.showProgress:
            bar.finish()

    def plot_melt_rates(self, vmin=0., vmax=50.):
        """
        Plot a series of image of the melt rate

        Parameters
        ----------
        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar
        """
        if 'timeMonthly_avg_landIceFreshwaterFlux' not in self.ds.keys():
            return
        rho_fw = 1000.
        secPerYear = 365 * 24 * 60 * 60

        da = (secPerYear / rho_fw *
              self.ds.timeMonthly_avg_landIceFreshwaterFlux)

        self.plot_horiz_series(da, 'melt rate', prefix='meltRate',
                               oceanDomain=False, units='m/yr', vmin=vmin,
                               vmax=vmax, cmap='cmo.amp')

    def plot_ice_shelf_boundary_variables(self):
        """
        Plot a series of image for each of several variables related to the
        ice shelf-ocean interface: heat flux from the ocean, heat flux into the
        ice, thermal driving, haline driving, and the friction velocity under
        ice
        """

        if 'timeMonthly_avg_landIceFreshwaterFlux' not in self.ds.keys():
            return
        self.plot_horiz_series(self.ds.timeMonthly_avg_landIceHeatFlux,
                               'heat flux from ocean to ice-ocean interface',
                               prefix='oceanHeatFlux',
                               oceanDomain=False, units='W/s',
                               vmin=-1e3, vmax=1e3, cmap='cmo.curl')

        self.plot_horiz_series(self.ds.timeMonthly_avg_heatFluxToLandIce,
                               'heat flux into ice at ice-ocean interface',
                               prefix='iceHeatFlux',
                               oceanDomain=False, units='W/s',
                               vmin=-1e1, vmax=1e1, cmap='cmo.curl')

        prefix = 'timeMonthly_avg_landIceBoundaryLayerTracers_'
        boundary_layer_temperature = \
            self.ds[f'{prefix}landIceBoundaryLayerTemperature']
        boundary_layer_salinity = \
            self.ds[f'{prefix}landIceBoundaryLayerSalinity']
        prefix = 'timeMonthly_avg_landIceInterfaceTracers_'
        interface_temperature = \
            self.ds[f'{prefix}landIceInterfaceTemperature']
        interface_salinity = \
            self.ds[f'{prefix}landIceInterfaceSalinity']
        da = boundary_layer_temperature - interface_temperature
        self.plot_horiz_series(da, 'thermal driving',
                               prefix='thermalDriving',
                               oceanDomain=False, units='deg C',
                               vmin=-2, vmax=2, cmap='cmo.curl')

        da = boundary_layer_salinity - interface_salinity
        self.plot_horiz_series(da, 'haline driving',
                               prefix='halineDriving',
                               oceanDomain=False, units='PSU',
                               vmin=-10, vmax=10, cmap='cmo.haline')

        self.plot_horiz_series(self.ds.timeMonthly_avg_landIceFrictionVelocity,
                               'friction velocity',
                               prefix='frictionVelocity',
                               oceanDomain=False, units='m/s',
                               vmin=1.e-4, vmax=1.e-1, cmap='cmo.speed',
                               cmap_scale='log')

    def plot_temperature(self):
        """
        Plot a series of images of temperature at the sea surface or
        ice-ocean interface, sea floor and in an x-z section
        """

        da = self.ds.timeMonthly_avg_activeTracers_temperature
        self.plot_3d_field_top_bot_section(da,
                                           nameInTitle='temperature',
                                           prefix='Temp', units='deg C',
                                           vmin=-2.5, vmax=1.0,
                                           cmap='cmo.thermal')

    def plot_salinity(self):
        """
        Plot a series of images of salinity at the sea surface or
        ice-ocean interface, sea floor and in an x-z section
        """

        da = self.ds.timeMonthly_avg_activeTracers_salinity
        self.plot_3d_field_top_bot_section(da,
                                           nameInTitle='salinity',
                                           prefix='Salinity', units='PSU',
                                           vmin=1., vmax=34.6,
                                           cmap='cmo.haline', cmap_scale='log')

    def plot_potential_density(self):
        """
        Plot a series of images of salinity at the sea surface or
        ice-ocean interface, sea floor and in an x-z section
        """

        da = self.ds.timeMonthly_avg_potentialDensity
        self.plot_3d_field_top_bot_section(da,
                                           nameInTitle='potential density',
                                           prefix='PotRho', units='kg/m^3',
                                           vmin=1027., vmax=1028.,
                                           cmap='cmo.dense')

    def plot_haney_number(self, haneyFolder=None):
        """
        Plot a series of images of the Haney number rx1 at the sea surface or
        ice-ocean interface, sea floor and in an x-z section

        Parameters
        ----------
        haneyFolder : str, optional
            The location of the haney number output, default is ``inFolder``
        """

        if haneyFolder is None:
            haneyFolder = self.inFolder
        ds = xarray.open_dataset('{}/haney.nc'.format(haneyFolder))

        self.plot_3d_field_top_bot_section(ds.haneyCell,
                                           nameInTitle='Haney Number (rx1)',
                                           prefix='Haney', units=None,
                                           vmin=0., vmax=8., cmap='cmo.matter')

    def plot_horiz_series(self, da, nameInTitle, prefix, oceanDomain,
                          units=None, vmin=None, vmax=None, cmap=None,
                          cmap_set_under=None, cmap_set_over=None,
                          cmap_scale='linear', time_indices=None,
                          figsize=(9, 3), contour_field=None):
        """
        Plot a series of image of a given variable

        Parameters
        ----------
        da : ``xarray.DataArray``
            The data array of time series to plot

        nameInTitle : str
            The name of the variable to use in the title and the progress bar

        prefix : str
            The nae of the variable to use in the subfolder and file prefix

        oceanDomain : bool
            True if the variable is for the full ocean, False if only for the
            cavity

        units : str, optional
            The units of the variable to be included in the title

        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar

        cmap : Colormap or str, optional
            A color map to plot

        cmap_set_under : str or None, optional
            A color for low out-of-range values

        cmap_scale : {'log', 'linear'}, optional
            Whether the colormap is logarithmic or linear

        time_indices : list of int, optional
            The time indices at which to plot. If not provided, set to all.
        """

        if 'Time' not in da.dims:
            da = da.expand_dims(dim='Time', axis=0)
        nTime = da.sizes['Time']
        if self.showProgress:
            widgets = ['plotting {}: '.format(nameInTitle),
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=nTime).start()
        else:
            bar = None

        if time_indices is None:
            time_indices = range(nTime)
        for tIndex in time_indices:
            self.update_date(tIndex)
            field = da.isel(Time=tIndex).values
            if contour_field is not None:
                contour_field = contour_field.isel(Time=tIndex).values
            outFileName = '{}/{}/{}_{:04d}.png'.format(
                self.outFolder, prefix, prefix, tIndex + 1)
            if units is None:
                title = nameInTitle
            else:
                title = '{} ({})'.format(nameInTitle, units)
            self._plot_horiz_field(field, title=title, outFileName=outFileName,
                                   oceanDomain=oceanDomain, vmin=vmin,
                                   vmax=vmax, cmap=cmap,
                                   cmap_set_under=cmap_set_under,
                                   cmap_set_over=cmap_set_over,
                                   cmap_scale=cmap_scale, figsize=figsize,
                                   contour_field=contour_field)
            if self.showProgress:
                bar.update(tIndex + 1)
        if self.showProgress:
            bar.finish()

    def plot_3d_field_top_bot_section(self, da, nameInTitle, prefix,
                                      units=None, vmin=None, vmax=None,
                                      cmap=None, cmap_scale='linear',
                                      cmap_set_under=None,
                                      cmap_set_over=None):
        """
        Plot a series of images of a given 3D variable showing the value
        at the top (sea surface or ice-ocean interface), sea floor and in an
        x-z section

        Parameters
        ----------
        da : ``xarray.DataArray``
            The data array of time series to plot

        nameInTitle : str
            The name of the variable to use in the title and the progress bar

        prefix : str
            The nae of the variable to use in the subfolder and file prefix

        units : str, optional
            The units of the variable to be included in the title

        vmin, vmax : float, optional
            The minimum and maximum values for the colorbar

        cmap : Colormap or str, optional
            A color map to plot

        cmap_set_under : str or None, optional
            A color for low out-of-range values
        """

        if vmin is None:
            vmin = da.min()
        if vmax is None:
            vmax = da.max()

        minLevelCell = self.dsMesh.minLevelCell - 1

        daTop = xarray.DataArray(da)

        daTop.coords['verticalIndex'] = \
            ('nVertLevels',
             numpy.arange(daTop.sizes['nVertLevels']))

        # mask only the values with the right vertical index
        daTop = daTop.where(daTop.verticalIndex == minLevelCell)

        # Each vertical layer has at most one non-NaN value so the "sum"
        # over the vertical is used to collapse the array in the vertical
        # dimension
        daTop = daTop.sum(dim='nVertLevels').where(minLevelCell >= 0)

        self.plot_horiz_series(daTop,
                               'top {}'.format(nameInTitle),
                               'top{}'.format(prefix), oceanDomain=True,
                               vmin=vmin, vmax=vmax, cmap=cmap,
                               cmap_scale=cmap_scale,
                               cmap_set_under=cmap_set_under,
                               cmap_set_over=cmap_set_over)

        maxLevelCell = self.dsMesh.maxLevelCell - 1

        daBot = xarray.DataArray(da)

        daBot.coords['verticalIndex'] = \
            ('nVertLevels',
             numpy.arange(daBot.sizes['nVertLevels']))

        # mask only the values with the right vertical index
        daBot = daBot.where(daBot.verticalIndex == maxLevelCell)

        # Each vertical layer has at most one non-NaN value so the "sum"
        # over the vertical is used to collapse the array in the vertical
        # dimension
        daBot = daBot.sum(dim='nVertLevels').where(maxLevelCell >= 0)

        self.plot_horiz_series(daBot,
                               'bot {}'.format(nameInTitle),
                               'bot{}'.format(prefix), oceanDomain=True,
                               vmin=vmin, vmax=vmax, cmap_scale=cmap_scale,
                               cmap=cmap)

        daSection = da.isel(nCells=self.sectionCellIndices)

        nTime = self.ds.sizes['Time']
        if self.showProgress:
            widgets = ['plotting {} section: '.format(nameInTitle),
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=nTime).start()
        else:
            bar = None

        for tIndex in range(nTime):
            self.update_date(tIndex)
            mask = numpy.logical_not(self.sectionMask)
            field = numpy.ma.masked_array(daSection.isel(Time=tIndex).values.T,
                                          mask=mask)
            outFileName = '{}/section{}/section{}_{:04d}.png'.format(
                self.outFolder, prefix, prefix, tIndex + 1)
            if units is None:
                title = nameInTitle
            else:
                title = '{} ({}) along section at y={:g} km'.format(
                    nameInTitle, units, 1e-3 * self.sectionY)
            self._plot_vert_field(self.X, self.Z[tIndex, :, :],
                                  field, title=title,
                                  outFileName=outFileName,
                                  vmin=vmin, vmax=vmax, cmap=cmap,
                                  show_boundaries=False, cmap_scale=cmap_scale)
            if self.showProgress:
                bar.update(tIndex + 1)
        if self.showProgress:
            bar.finish()

    def plot_layer_interfaces(self, figsize=(9, 5)):
        """
        Plot layer interfaces, the sea surface height and the bottom topography
        of the cross section at fixed y.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure
        """

        nTime = self.Z.shape[0]

        if self.showProgress:
            widgets = ['plotting section of layer interfaces: ',
                       progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
            bar = progressbar.ProgressBar(widgets=widgets,
                                          maxval=nTime).start()
        else:
            bar = None

        z_mask = numpy.ones(self.X.shape)
        z_mask[0:-1, 0:-1] *= numpy.where(self.sectionMask, 1., numpy.nan)
        z_mask[1:, 0:-1] *= numpy.where(self.sectionMask, 1., numpy.nan)
        z_mask[0:-1, 1:] *= numpy.where(self.sectionMask, 1., numpy.nan)
        z_mask[1:, 1:] *= numpy.where(self.sectionMask, 1., numpy.nan)

        for tIndex in range(nTime):
            Z = numpy.array(self.Z[tIndex, :, :])
            ylim = [numpy.amin(Z), 20]
            Z *= z_mask
            X = self.X
            self.update_date(tIndex)

            outFileName = '{}/layers/layers_{:04d}.png'.format(self.outFolder,
                                                               tIndex + 1)

            if os.path.exists(outFileName):
                continue

            try:
                os.makedirs(os.path.dirname(outFileName))
            except OSError:
                pass

            plt.figure(figsize=figsize)
            ax = plt.subplot(111)

            for z_index in range(1, X.shape[0]):
                plt.plot(1e-3 * X[z_index, :], Z[z_index, :], 'k')
            plt.plot(1e-3 * X[0, :], Z[0, :], 'b')
            plt.plot(1e-3 * X[0, :], self.zBotSection, 'g')
            plt.plot(1e-3 * X[0, :], self.landIceDraft, 'r')

            ax.autoscale(tight=True)
            x1, x2, y1, y2 = 420, 470, -650, -520
            xlim = [min(x1, 1e-3 * numpy.amin(X)), 1e-3 * numpy.amax(X)]
            plt.xlim(xlim)
            plt.ylim(ylim)
            axins = ax.inset_axes([0.01, 0.6, 0.3, 0.39])
            for z_index in range(1, X.shape[0]):
                axins.plot(1e-3 * X[z_index, :], Z[z_index, :], 'k')
            axins.plot(1e-3 * X[0, :], Z[0, :], 'b')
            axins.plot(1e-3 * X[0, :], self.zBotSection, 'g')
            axins.plot(1e-3 * X[0, :], self.landIceDraft, 'r')
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            ax.indicate_inset_zoom(axins, edgecolor="black")
            plt.title('{} {}'.format('layer interfaces', self.date))
            plt.tight_layout(pad=0.5)
            plt.savefig(outFileName)
            plt.close()

            if self.showProgress:
                bar.update(tIndex + 1)
        if self.showProgress:
            bar.finish()

    def images_to_movies(self, outFolder, framesPerSecond=30, extension='mp4',
                         overwrite=True):
        """
        Convert all the image sequences into movies with ffmpeg
        """
        try:
            os.makedirs('{}/logs'.format(outFolder))
        except OSError:
            pass

        framesPerSecond = '{}'.format(framesPerSecond)

        for fileName in sorted(glob.glob(
                '{}/*/*0001.png'.format(self.outFolder))):
            prefix = os.path.basename(fileName)[:-9]
            outFileName = '{}/{}.{}'.format(outFolder, prefix, extension)
            if not overwrite and os.path.exists(outFileName):
                continue

            imageFileTemplate = '{}/{}/{}_%04d.png'.format(self.outFolder,
                                                           prefix, prefix)
            logFileName = '{}/logs/{}.log'.format(outFolder, prefix)
            with open(logFileName, 'w') as logFile:
                args = ['ffmpeg', '-y', '-r', framesPerSecond,
                        '-i', imageFileTemplate, '-b:v', '32000k',
                        '-r', framesPerSecond, '-pix_fmt', 'yuv420p',
                        outFileName]
                print('running {}'.format(' '.join(args)))
                subprocess.check_call(args, stdout=logFile, stderr=logFile)

    def update_date(self, tIndex):
        if 'xtime_startMonthly' in self.ds:
            var = 'xtime_startMonthly'
        elif 'xtime' in self.ds:
            var = 'xtime'
        else:
            self.date = ''
            return

        xtime = self.ds[var].isel(Time=tIndex).values
        xtime = ''.join(str(xtime.astype('U'))).strip()
        year = xtime[0:4]
        month = xtime[5:7]
        day = xtime[8:10]
        self.date = '{}-{}-{}'.format(year, month, day)

    def _plot_horiz_field(self, field, title, outFileName, oceanDomain=True,
                          vmin=None, vmax=None, figsize=(9, 3), cmap=None,
                          cmap_set_under=None, cmap_set_over=None,
                          cmap_scale='linear', contour_field=None):

        try:
            os.makedirs(os.path.dirname(outFileName))
        except OSError:
            pass

        if os.path.exists(outFileName):
            return

        if oceanDomain:
            localPatches = copy.deepcopy(self.oceanPatches)
            localPatches.set_array(field[self.oceanMask])
        else:
            localPatches = copy.deepcopy(self.cavityPatches)
            localPatches.set_array(field[self.cavityMask])

        if cmap is not None:
            localPatches.set_cmap(cmap)
        if cmap_set_under is not None:
            current_cmap = localPatches.get_cmap()
            current_cmap.set_under(cmap_set_under)
        if cmap_set_over is not None:
            current_cmap = localPatches.get_cmap()
            current_cmap.set_over(cmap_set_over)
        localPatches.set_edgecolor('face')
        localPatches.set_clim(vmin=vmin, vmax=vmax)

        if cmap_scale == 'log':
            localPatches.set_norm(LogNorm(vmin=max(1e-10, vmin),
                                  vmax=vmax, clip=False))
        elif cmap_scale == 'symlog':
            localPatches.set_norm(SymLogNorm(vmin=vmin, vmax=vmax, clip=False,
                                             linthresh=vmax / 1e2))

        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.add_collection(localPatches)
        if contour_field is not None:
            dsMesh = self.dsMesh
            ax.tricontour(dsMesh.xCell / 1.e3, dsMesh.yCell / 1.e3,
                          contour_field,
                          colors='k', levels=[1.1e-2 + 1.e-8])
            ax.tricontour(dsMesh.xCell / 1.e3, dsMesh.yCell / 1.e3,
                          contour_field,
                          colors='grey', levels=numpy.arange(50., 700., 50.))
        plt.colorbar(localPatches, extend='both')
        plt.axis([0, 500, 0, 1000])
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        plt.title('{} {}'.format(title, self.date))
        plt.tight_layout(pad=0.5)
        plt.savefig(outFileName)
        plt.close()

    def _plot_vert_field(self, inX, inZ, field, title, outFileName,
                         vmin=None, vmax=None, figsize=(9, 5), cmap=None,
                         show_boundaries=True, cmap_scale='linear'):
        try:
            os.makedirs(os.path.dirname(outFileName))
        except OSError:
            pass

        if os.path.exists(outFileName):
            return

        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        z_mask = numpy.ones(self.X.shape)
        z_mask[0:-1, 0:-1] *= numpy.where(self.sectionMask, 1., numpy.nan)

        tIndex = 0
        Z = numpy.array(self.Z[tIndex, :, :])
        Z *= z_mask
        X = self.X

        plt.fill_between(1e-3 * X[0, :], self.zBotSection, y2=0,
                         facecolor='lightsteelblue', zorder=2)
        plt.fill_between(1e-3 * X[0, :], self.zBotSection, y2=-750,
                         facecolor='grey', zorder=1)
        if show_boundaries:
            for z_index in range(1, X.shape[0]):
                plt.plot(1e-3 * X[z_index, :], Z[z_index, :], 'k', zorder=4)
        plt.pcolormesh(1e-3 * inX, inZ, field, vmin=vmin, vmax=vmax, cmap=cmap,
                       zorder=3)
        plt.colorbar()
        x1, x2, y1, y2 = 455, 470, -640, -540
        axins = ax.inset_axes([0.01, 0.6, 0.3, 0.39])
        axins.fill_between(1e-3 * X[0, :], self.zBotSection, y2=0,
                           facecolor='lightsteelblue', zorder=2)
        axins.fill_between(1e-3 * X[0, :], self.zBotSection, y2=-750,
                           facecolor='grey', zorder=1)
        if show_boundaries:
            for z_index in range(1, X.shape[0]):
                axins.plot(1e-3 * X[z_index, :], Z[z_index, :], 'k', zorder=4)
        axins.pcolormesh(1e-3 * inX, inZ, field, vmin=vmin, vmax=vmax,
                         cmap=cmap, zorder=3)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        ax.indicate_inset_zoom(axins, edgecolor="black")
        ax.autoscale(tight=True)
        plt.ylim([numpy.amin(inZ), 20])
        plt.xlim([400, 800])
        plt.title('{} {}'.format(title, self.date))
        plt.tight_layout(pad=0.5)
        plt.savefig(outFileName, dpi='figure')
        plt.close()

    def _compute_section_x_z(self):
        if 'xIsomipCell' in self.dsMesh.keys():
            x = _interp_extrap_corner(
                self.dsMesh.xIsomipCell[self.sectionCellIndices])
        else:
            x = _interp_extrap_corner(
                self.dsMesh.xCell[self.sectionCellIndices])
        nx = len(x)
        nVertLevels = self.dsMesh.sizes['nVertLevels']
        nTime = self.ds.sizes['Time']
        self.X = numpy.zeros((nVertLevels + 1, nx))
        for zIndex in range(nVertLevels + 1):
            self.X[zIndex, :] = x

        self.sectionMask = numpy.zeros((nVertLevels, nx - 1), dtype=bool)
        for zIndex in range(nVertLevels):
            minLevelCell = self.dsMesh.minLevelCell.isel(
                nCells=self.sectionCellIndices) - 1
            maxLevelCell = self.dsMesh.maxLevelCell.isel(
                nCells=self.sectionCellIndices) - 1
            self.sectionMask[zIndex, :] = numpy.logical_and(
                zIndex >= minLevelCell, zIndex <= maxLevelCell)

        self.Z = numpy.zeros((nTime, nVertLevels + 1, nx))
        self.zBotSection = -_interp_extrap_corner(
            self.dsMesh.bottomDepth.isel(
                nCells=self.sectionCellIndices).values)
        for tIndex in range(nTime):
            if 'timeMonthly_avg_layerThickness' in self.ds:
                var = 'timeMonthly_avg_layerThickness'
            else:
                var = 'layerThickness'
            layerThickness = self.ds[var].isel(
                Time=tIndex, nCells=self.sectionCellIndices)
            layerThickness = layerThickness.values * self.sectionMask.T
            layerThickness = numpy.nan_to_num(layerThickness)
            self.Z[tIndex, -1, :] = self.zBotSection
            for zIndex in range(nVertLevels - 1, -1, -1):
                layerThicknessSection = _interp_extrap_corner(
                    layerThickness[:, zIndex])
                self.Z[tIndex, zIndex, :] = self.Z[tIndex, zIndex + 1, :] + \
                    layerThicknessSection
            if 'timeMonthly_avg_landIceDraft' in self.ds:
                var = 'timeMonthly_avg_landIceDraft'
            else:
                var = 'landIceDraft'
            landIceDraft = self.ds[var].isel(
                Time=tIndex, nCells=self.sectionCellIndices)
            landIceDraft = landIceDraft.values * self.sectionMask[0, :].T
            landIceDraft = numpy.nan_to_num(landIceDraft)
            self.landIceDraft = _interp_extrap_corner(landIceDraft)


def _compute_cell_patches(dsMesh, mask):
    patches = []
    nVerticesOnCell = dsMesh.nEdgesOnCell.values
    verticesOnCell = dsMesh.verticesOnCell.values - 1
    if 'xIsomipVertex' in dsMesh.keys():
        xVertex = dsMesh.xIsomipVertex.values
        yVertex = dsMesh.yIsomipVertex.values
    else:
        xVertex = dsMesh.xVertex.values
        yVertex = dsMesh.yVertex.values
    for iCell in range(dsMesh.sizes['nCells']):
        if not mask[iCell]:
            continue
        nVert = nVerticesOnCell[iCell]
        vertexIndices = verticesOnCell[iCell, :nVert]
        vertices = numpy.zeros((nVert, 2))
        vertices[:, 0] = 1e-3 * xVertex[vertexIndices]
        vertices[:, 1] = 1e-3 * yVertex[vertexIndices]

        polygon = Polygon(vertices, closed=True)
        patches.append(polygon)

    p = PatchCollection(patches, alpha=1.)

    return p


def _compute_section_cell_indices(y, dsMesh):
    if 'xIsomipCell' in dsMesh.keys():
        xCell = dsMesh.xIsomipCell.values
        yCell = dsMesh.yIsomipCell.values
    else:
        xCell = dsMesh.xCell.values
        yCell = dsMesh.yCell.values
    xMin = numpy.amin(xCell)
    xMax = numpy.amax(xCell)
    xs = numpy.linspace(xMin, xMax, 10000)
    cellIndices = []
    for x in xs:
        distanceSquared = (x - xCell)**2 + (y - yCell)**2
        index = numpy.argmin(distanceSquared)
        if len(cellIndices) == 0 or cellIndices[-1] != index:
            cellIndices.append(index)

    return numpy.array(cellIndices)


def _interp_extrap_corner(inField):
    """Interpolate/extrapolate a 1D field from grid centers to grid corners"""

    outField = numpy.zeros(len(inField) + 1)
    outField[1:-1] = 0.5 * (inField[0:-1] + inField[1:])
    # extrapolate the ends
    outField[0] = 1.5 * inField[0] - 0.5 * inField[1]
    outField[-1] = 1.5 * inField[-1] - 0.5 * inField[-2]
    return outField
