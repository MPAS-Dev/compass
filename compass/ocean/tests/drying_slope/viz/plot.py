#!/usr/bin/env python
"""

Drying slope comparison betewen MPAS-O, analytical, and ROMS results from

Warner, J. C., Defne, Z., Haas, K., & Arango, H. G. (2013). A wetting and
drying scheme for ROMS. Computers & geosciences, 58, 54-61.

Phillip J. Wolfram and Zhendong Cao
04/30/2019

"""

import os
import numpy
import xarray
import matplotlib.pyplot as plt
import pandas as pd
import subprocess


class TimeSeriesPlotter(object):
    """
    A plotter object to hold on to some info needed for plotting time series
    from drying slope simulation results

    Attributes
    ----------
    inFolder : str
        The folder with simulation results

    outFolder : str
        The folder where images will be written

    """
    def __init__(self, inFolder='.', outFolder='plots'):
        """
        Create a plotter object to hold on to some info needed for plotting
        time series from drying slope simulation results

        Parameters
        ----------
        inFolder : str, optional
            The folder with simulation results

        outFolder : str, optional
            The folder where images will be written

        dsMesh : xarray.Dataset, optional
            The MPAS mesh

        ds : xarray.Dataset, optional
            The time series output
        """

        self.inFolder = inFolder
        self.outFolder = outFolder

        if not os.path.exists(outFolder):
            try:
                os.makedirs(os.path.join(os.getcwd(), self.outFolder))
            except OSError:
                pass

        plt.switch_backend('Agg')

    def plot_ssh_time_series(self):
        """
        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}
        xSsh = numpy.linspace(0, 12.0, 100)
        ySsh = 10.0*numpy.sin(xSsh*numpy.pi/12.0) - 10.0

        figsize = [6.4, 4.8]
        markersize = 20

        # SSH forcing figure
        damping_coeffs = [0.0025, 0.01]
        fig1, ax1 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                 figsize=figsize, dpi=100)
        for i, damping_coeff in enumerate(damping_coeffs):
            ds = xarray.open_dataset('output_{}.nc'.format(damping_coeff))
            ssh = ds.ssh
            ympas = ds.ssh.where(ds.tidalInputMask).mean('nCells').values
            xmpas = numpy.linspace(0, 1.0, len(ds.xtime))*12.0
            ax1[i].plot(xmpas, ympas, marker='o', label='MPAS-O forward',
                        color=colors['MPAS-O'])
            ax1[i].plot(xSsh, ySsh, lw=3, label='analytical',
                        color=colors['analytical'])
            ax1[i].set_ylabel('Tidal amplitude (m)')
            ds.close()

        ax1[0].legend(frameon=False)
        ax1[1].set_xlabel('Time (hrs)')

        fig1.suptitle('Tidal amplitude forcing (right side)')
        fig1.savefig('{}/ssh_t.png'.format(self.outFolder),
                     bbox_inches='tight', dpi=200)
        plt.close(fig1)

    def plot_ssh_validation(self):
        """
        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}

        # Cross-section figure
        times = ['0.50', '0.05', '0.40', '0.15', '0.30', '0.25']
        locs = [9.3, 7.2, 4.2, 2.2, 1.2, 0.2]
        locs = 0.92 - numpy.divide(locs, 11.)

        damping_coeffs = [0.0025, 0.01]

        fig2, ax2 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                 sharex=True, sharey=True)
        fig2.text(0.04, 0.5, 'Channel depth (m)', va='center',
                  rotation='vertical')
        fig2.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

        xBed = numpy.linspace(0, 25, 100)
        yBed = 10.0/25.0*xBed

        for i, damping_coeff in enumerate(damping_coeffs):
            ds = xarray.open_dataset('output_{}.nc'.format(damping_coeff))
            ds = ds.drop_vars(numpy.setdiff1d([j for j in ds.variables],
                                              ['yCell', 'ssh']))

            ax2[i].plot(xBed, yBed, '-k', lw=3)
            ax2[i].text(0.5, 5, 'r = ' + str(damping_coeff))
            ax2[i].set_xlim(0, 25)
            ax2[i].set_ylim(-1, 11)
            ax2[i].invert_yaxis()
            ax2[i].spines['top'].set_visible(False)
            ax2[i].spines['right'].set_visible(False)

            for atime, ay in zip(times, locs):

                # Plot MPAS-O data
                # factor of 1e- needed to account for annoying round-off issue
                # to get right time slices
                plottime = int((float(atime)/0.2 + 1e-16)*24.0)
                ymean = ds.isel(Time=plottime).groupby('yCell').mean(
                                dim=xarray.ALL_DIMS)
                x = ymean.yCell.values/1000.0
                y = ymean.ssh.values

                mpas = ax2[i].plot(x, -y, label='MPAS-O',
                                   color=colors['MPAS-O'])
                ax2[i].text(1, ay, atime + ' days', size=8,
                            transform=ax2[i].transAxes)

                # Plot comparison data
                for datatype in ['analytical', 'ROMS']:
                    datafile = './r{}d{}-{}.csv'.format(
                                damping_coeff, atime,
                                datatype.lower())
                    data = pd.read_csv(datafile, header=None)
                    ax2[i].scatter(data[0], data[1], marker='.',
                                   color=colors[datatype], label=datatype)

            ds.close()

        h, l0 = ax2[0].get_legend_handles_labels()
        ax2[0].legend(h[0:3], l0[0:3], frameon=False, loc='lower left')
        h, l1 = ax2[1].get_legend_handles_labels()
        ax2[1].legend(h[0:3], l1[0:3], frameon=False, loc='lower left')

        if not os.path.exists(self.outFolder):
            try:
                os.makedirs(os.path.join(os.getcwd(), self.outFolder))
            except OSError:
                pass

        fig2.savefig('{}/ssh_depth_section.png'.format(self.outFolder),
                     dpi=200)
        plt.close(fig2)


class MoviePlotter(object):
    """
    A plotter object to hold on to some info needed for plotting images from
    drying slope simulation results

    Attributes
    ----------
    inFolder : str
        The folder with simulation results

    outFolder : str
        The folder where images will be written
    """

    def __init__(self, inFolder='.', outFolder='plots'):
        """
        Create a plotter object to hold on to some info needed for plotting
        images from drying slope simulation results

        Parameters
        ----------
        inFolder : str
            The folder with simulation results

        outFolder : str
            The folder where images will be written
        """
        self.inFolder = inFolder
        self.outFolder = outFolder

        if not os.path.exists(outFolder):
            try:
                os.makedirs(os.path.join(os.getcwd(), self.outFolder))
            except OSError:
                pass

        plt.switch_backend('Agg')

    def plot_ssh_validation(self):
        """
        Compare ssh along the channel at different time slices with the
        analytical solution and ROMS results.

        Parameters
        ----------

        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}

        times = ['0.50', '0.05', '0.40', '0.15', '0.30', '0.25']
        locs = [9.3, 7.2, 4.2, 2.2, 1.2, 0.2]
        locs = 0.92 - numpy.divide(locs, 11.)

        damping_coeffs = [0.0025, 0.01]

        xBed = numpy.linspace(0, 25, 100)
        yBed = 10.0/25.0*xBed

        ii = 0
        # Plot profiles over the 12h simulation duration
        for itime in numpy.linspace(0, 0.5, 5*12+1):

            plottime = int((float(itime)/0.2 + 1e-16)*24.0)

            fig2, ax2 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                     sharex=True, sharey=True)
            ax2[0].set_title('t = {0:.3f} days'.format(itime))
            fig2.text(0.04, 0.5, 'Channel depth (m)', va='center',
                      rotation='vertical')
            fig2.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

            for i, damping_coeff in enumerate(damping_coeffs):

                ds = xarray.open_dataset('output_{}.nc'.format(damping_coeff))
                ds = ds.drop_vars(numpy.setdiff1d([j for j in ds.variables],
                                                  ['yCell', 'ssh']))

                # Plot MPAS-O snapshots
                # factor of 1e- needed to account for annoying round-off issue
                # to get right time slices
                ymean = ds.isel(Time=plottime).groupby('yCell').mean(
                                dim=xarray.ALL_DIMS)
                x = ymean.yCell.values/1000.0
                y = ymean.ssh.values
                ax2[i].plot(xBed, yBed, '-k', lw=3)
                mpas = ax2[i].plot(x, -y, label='MPAS-O',
                                   color=colors['MPAS-O'])

                ax2[i].text(0.5, 5, 'r = ' + str(damping_coeff))
                ax2[i].set_ylim(-1, 11)
                ax2[i].set_xlim(0, 25)
                ax2[i].invert_yaxis()
                ax2[i].spines['top'].set_visible(False)
                ax2[i].spines['right'].set_visible(False)

                # Plot comparison data
                for atime, ay in zip(times, locs):
                    ax2[i].text(1, ay, atime + ' days', size=8,
                                transform=ax2[i].transAxes)

                    for datatype in ['analytical', 'ROMS']:
                        datafile = './r{}d{}-{}.csv'.format(
                                    damping_coeff, atime,
                                    datatype.lower())
                        data = pd.read_csv(datafile, header=None)
                        ax2[i].scatter(data[0], data[1], marker='.',
                                       color=colors[datatype], label=datatype)

                ds.close()

            h, l0 = ax2[0].get_legend_handles_labels()
            ax2[0].legend(h[0:3], l0[0:3], frameon=False, loc='lower left')
            h, l1 = ax2[1].get_legend_handles_labels()
            ax2[1].legend(h[0:3], l1[0:3], frameon=False, loc='lower left')

            fig2.savefig('{}/ssh_depth_section_{:03d}.png'.format(
                         self.outFolder, ii), dpi=200)
            plt.close(fig2)
            ii += 1

    def images_to_movies(self, outFolder='.', framesPerSecond=30,
                         extension='mp4', overwrite=True):
        """
        Convert all the image sequences into movies with ffmpeg
        """
        try:
            os.makedirs('{}/logs'.format(outFolder))
        except OSError:
            pass

        framesPerSecond = '{}'.format(framesPerSecond)
        prefix = 'ssh_depth_section'
        outFileName = '{}/{}.{}'.format(outFolder, prefix, extension)
        if overwrite or not os.path.exists(outFileName):

            imageFileTemplate = '{}/{}_%03d.png'.format(self.outFolder,
                                                        prefix)
            logFileName = '{}/logs/{}.log'.format(outFolder, prefix)
            with open(logFileName, 'w') as logFile:
                args = ['ffmpeg', '-y', '-r', framesPerSecond,
                        '-i', imageFileTemplate, '-b:v', '32000k',
                        '-r', framesPerSecond, '-pix_fmt', 'yuv420p',
                        outFileName]
                print('running {}'.format(' '.join(args)))
                subprocess.check_call(args, stdout=logFile, stderr=logFile)
