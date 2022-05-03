import os
import xarray
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing drying slope results, as well as comparison with
    analytical solution and ROMS results.

    Attributes
    ----------
    damping_coeffs : list of float
        Rayleigh damping coefficients used for the MPAS-Ocean and ROMS
        forward runs

    times : list of str
        Time in days since the start of the simulation for which ROMS
        comparison data is available

    datatypes : list of str
        The sources of data for comparison to the MPAS-Ocean run
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        damping_coeffs = [0.0025, 0.01]
        times = ['0.05', '0.15', '0.25', '0.30', '0.40', '0.50']
        datatypes = ['analytical', 'ROMS']
        self.damping_coeffs = damping_coeffs
        self.times = times
        self.datatypes = datatypes

        for damping_coeff in damping_coeffs:
            self.add_input_file(filename=f'output_{damping_coeff}.nc',
                                target=f'../forward_{damping_coeff}/output.nc')
            for time in times:
                for datatype in datatypes:
                    filename = f'r{damping_coeff}d{time}-{datatype.lower()}'\
                               '.csv'
                    self.add_input_file(filename=filename, target=filename,
                                        database='drying_slope')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['paths']
        datapath = section.get('ocean_database_root')
        section = self.config['drying_slope_viz']
        generate_movie = section.getboolean('generate_movie')

        self._plot_ssh_validation()
        self._plot_ssh_time_series()
        if generate_movie:
            frames_per_second = section.getint('frames_per_second')
            movie_format = section.get('movie_format')
            outFolder = 'movie'
            if not os.path.exists(outFolder):
                try:
                    os.makedirs(os.path.join(os.getcwd(), outFolder))
                except OSError:
                    pass
            self._plot_ssh_validation_for_movie(outFolder=outFolder)
            self._images_to_movies(framesPerSecond=frames_per_second,
                                   outFolder=outFolder, extension=movie_format)

    def _plot_ssh_time_series(self, outFolder='.'):
        """
        Plot ssh forcing on the right x boundary as a function of time against
        the analytical solution. The agreement should be within machine
        precision if the namelist options are consistent with the Warner et al.
        (2013) test case.
        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}
        xSsh = numpy.linspace(0, 12.0, 100)
        ySsh = 10.0*numpy.sin(xSsh*numpy.pi/12.0) - 10.0

        figsize = [6.4, 4.8]
        markersize = 20

        damping_coeffs = self.damping_coeffs
        fig1, ax1 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                 figsize=figsize, dpi=100)
        for i, damping_coeff in enumerate(damping_coeffs):
            ds = xarray.open_dataset(f'output_{damping_coeff}.nc')
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
        fig1.savefig(f'{outFolder}/ssh_t.png', bbox_inches='tight', dpi=200)

        plt.close(fig1)

    def _plot_ssh_validation(self, outFolder='.'):
        """
        Plot ssh as a function of along-channel distance for all times for
        which there is validation data
        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}

        locs = [7.2, 2.2, 0.2, 1.2, 4.2, 9.3]
        locs = 0.92 - numpy.divide(locs, 11.)

        times = self.times
        damping_coeffs = self.damping_coeffs
        datatypes = self.datatypes

        fig2, ax2 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                 sharex=True, sharey=True)
        fig2.text(0.04, 0.5, 'Channel depth (m)', va='center',
                  rotation='vertical')
        fig2.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

        xBed = numpy.linspace(0, 25, 100)
        yBed = 10.0/25.0*xBed

        for i, damping_coeff in enumerate(damping_coeffs):
            ds = xarray.open_dataset(f'output_{damping_coeff}.nc')
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
                for datatype in datatypes:
                    datafile = f'./r{damping_coeff}d{atime}-'\
                               f'{datatype.lower()}.csv'
                    data = pd.read_csv(datafile, header=None)
                    ax2[i].scatter(data[0], data[1], marker='.',
                                   color=colors[datatype], label=datatype)

            ds.close()

        h, l0 = ax2[0].get_legend_handles_labels()
        ax2[0].legend(h[0:3], l0[0:3], frameon=False, loc='lower left')
        h, l1 = ax2[1].get_legend_handles_labels()
        ax2[1].legend(h[0:3], l1[0:3], frameon=False, loc='lower left')

        fig2.savefig(f'{outFolder}/ssh_depth_section.png', dpi=200)
        plt.close(fig2)

    def _plot_ssh_validation_for_movie(self, outFolder='.'):
        """
        Compare ssh along the channel at different time slices with the
        analytical solution and ROMS results.

        Parameters
        ----------

        """
        colors = {'MPAS-O': 'k', 'analytical': 'b', 'ROMS': 'g'}

        locs = [7.2, 2.2, 0.2, 1.2, 4.2, 9.3]
        locs = 0.92 - numpy.divide(locs, 11.)

        times = self.times
        damping_coeffs = self.damping_coeffs
        datatypes = self.datatypes

        xBed = numpy.linspace(0, 25, 100)
        yBed = 10.0/25.0*xBed

        ii = 0
        # Plot profiles over the 12h simulation duration
        for itime in numpy.linspace(0, 0.5, 5*12+1):

            plottime = int((float(itime)/0.2 + 1e-16)*24.0)

            fig2, ax2 = plt.subplots(nrows=len(damping_coeffs), ncols=1,
                                     sharex=True, sharey=True)
            ax2[0].set_title(f't = {itime:.3f} days')
            fig2.text(0.04, 0.5, 'Channel depth (m)', va='center',
                      rotation='vertical')
            fig2.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

            for i, damping_coeff in enumerate(damping_coeffs):

                ds = xarray.open_dataset(f'output_{damping_coeff}.nc')
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
                    ax2[i].text(1, ay, f'{atime} days', size=8,
                                transform=ax2[i].transAxes)

                    for datatype in datatypes:
                        datafile = f'./r{damping_coeff}d{atime}-'\
                                   f'{datatype.lower()}.csv'
                        data = pd.read_csv(datafile, header=None)
                        ax2[i].scatter(data[0], data[1], marker='.',
                                       color=colors[datatype], label=datatype)

                ds.close()

            h, l0 = ax2[0].get_legend_handles_labels()
            ax2[0].legend(h[0:3], l0[0:3], frameon=False, loc='lower left')
            h, l1 = ax2[1].get_legend_handles_labels()
            ax2[1].legend(h[0:3], l1[0:3], frameon=False, loc='lower left')

            fig2.savefig(f'{outFolder}/ssh_depth_section_{ii:03d}.png',
                         dpi=200)
            plt.close(fig2)
            ii += 1

    def _images_to_movies(self, outFolder='.', framesPerSecond=30,
                          extension='mp4', overwrite=True):
        """
        Convert all the image sequences into movies with ffmpeg
        """
        try:
            os.makedirs(f'{outFolder}/logs')
        except OSError:
            pass

        framesPerSecond = str(framesPerSecond)
        prefix = 'ssh_depth_section'
        outFileName = f'{outFolder}/{prefix}.{extension}'
        if overwrite or not os.path.exists(outFileName):

            imageFileTemplate = f'{outFolder}/{prefix}_%03d.png'
            logFileName = f'{outFolder}/logs/{prefix}.log'
            with open(logFileName, 'w') as logFile:
                args = ['ffmpeg', '-y', '-r', framesPerSecond,
                        '-i', imageFileTemplate, '-b:v', '32000k',
                        '-r', framesPerSecond, '-pix_fmt', 'yuv420p',
                        outFileName]
                print_args = ' '.join(args)
                print(f'running {print_args}')
                subprocess.check_call(args, stdout=logFile, stderr=logFile)
