import os
import xarray
import numpy
import matplotlib.pyplot as plt
import pandas as pd

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing XX
    """
    def __init__(self, test_case, config):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        for damping_coeff in [0.0025, 0.01]:
            self.add_input_file(filename='output_{}.nc'.format(damping_coeff),
                                target='../forward_{}/output.nc'.format(damping_coeff))
        self.add_output_file('ssh_t.png')
        self.add_output_file('ssh_depth_section.png')
        self.config = config

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['paths']
        datapath = section.get('ocean_database_root')

        colors = {'MPAS-O':'k', 'analytical':'b', 'ROMS':'g'}
        xSsh = numpy.linspace(0,12.0,100)
        ySsh = 10.0*numpy.sin(xSsh*numpy.pi/12.0) - 10.0

        figsize = [6.4, 4.8]
        markersize = 20

        # SSH forcing figure
        damping_coeffs = [0.0025, 0.01]
        fig1, ax1 = plt.subplots(nrows=len(damping_coeffs), ncols=1, figsize=figsize, dpi=100)

        # Cross-section figure
        times = ['0.50', '0.05', '0.40', '0.15', '0.30', '0.25']
        locs = [9.3, 7.2, 4.2, 2.2, 1.2, 0.2]
        locs = 0.92 - numpy.divide(locs, 11.)
        fig2, ax2 = plt.subplots(nrows=len(damping_coeffs), ncols=1, sharex=True, sharey=True)
        fig2.text(0.04, 0.5, 'Channel depth (m)', va='center', rotation='vertical')
        fig2.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

        #TODO replace with parameters
        xBed = numpy.linspace(0,25,100)
        yBed = 10.0/25.0*xBed

        for i, damping_coeff in enumerate(damping_coeffs):
            ds = xarray.open_dataset('output_{}.nc'.format(damping_coeff))
            ssh = ds.ssh
            ympas = ds.ssh.where(ds.tidalInputMask).mean('nCells').values
            xmpas = numpy.linspace(0, 1.0, len(ds.xtime))*12.0
            ax1[i].plot(xmpas, ympas, marker='o', label='MPAS-O forward', color=colors['MPAS-O'])
            ax1[i].plot(xSsh, ySsh, lw=3, color=colors['analytical'], label='analytical')
            ax1[i].set_ylabel('Tidal amplitude (m)')

            ax2[i].text(0.5, 5, 'r = ' + str(damping_coeff))
            ax2[i].set_ylim(-1, 11)
            ax2[i].invert_yaxis()
            ax2[i].spines['top'].set_visible(False)
            ax2[i].spines['right'].set_visible(False)
            for atime, ay in zip(times, locs):

                # factor of 1e- needed to account for annoying round-off issue to get right time slices
                plottime = int((float(atime)/0.2 + 1e-16)*24.0)
                ds = ds.drop_vars(numpy.setdiff1d([i for i in ds.variables], ['yCell','ssh']))
                ymean = ds.isel(Time=plottime).groupby('yCell').mean(dim=xarray.ALL_DIMS)
                x = ymean.yCell.values/1000.0
                y = ymean.ssh.values

                ax2[i].plot(xBed, yBed, '-k', lw=3)
                mpas = ax2[i].plot(x, -y, color=colors['MPAS-O'], label='MPAS-O')
                ax2[i].text(1, ay, atime + ' days', size=8, transform=ax2[i].transAxes)

                for datatype in ['analytical', 'ROMS']:
                    datafile = '{}/drying_slope/r{}d{}-{}.csv'.format(
                                datapath,damping_coeff,atime,datatype.lower())
                    data = pd.read_csv(datafile, header=None)
                    ax2[i].scatter(data[0], data[1],
                                  marker = '.', color = colors[datatype], label=datatype)

            ds.close()

        ax1[0].legend(frameon=False)
        ax1[1].set_xlabel('Time (hrs)')

        h, l = ax2[0].get_legend_handles_labels()
        ax2[0].legend(h[1:4], l[1:4], frameon=False, loc='lower left')
        h, l = ax2[1].get_legend_handles_labels()
        ax2[1].legend(h[1:4], l[1:4], frameon=False, loc='lower left')
        ax2[1].set_xlim(0,25)

        fig1.suptitle('Tidal amplitude forcing (right side) for MPAS-O and analytical')
        fig1.savefig('ssh_t.png', bbox_inches='tight', dpi=200)
        plt.close(fig1)

        fig2.savefig('ssh_depth_section.png', dpi=200)
        plt.close(fig2)

