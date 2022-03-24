import xarray
import numpy
import matplotlib.pyplot as plt

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing XX
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

        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        self.add_output_file('ssh_t.png')
        self.add_output_file('ssh_depth_section.png')

    def run(self):
        """
        Run this step of the test case
        """

        ds = xarray.open_dataset('output.nc')
        ssh = ds.ssh
        ympas = ds.ssh.where(ds.tidalInputMask).mean('nCells').values
        xmpas = numpy.linspace(0, 1.0, len(ds.xtime))*12.0
        x = numpy.linspace(0,12.0,100)
        y = 10.0*numpy.sin(x*numpy.pi/12.0) - 10.0
        print(ympas)
        #print('ymin={} ymax={}\n'.format(ympas.min(), ympas.max()))

        figsize = [6.4, 4.8]
        markersize = 20
        # Figures
        plt.figure(figsize=figsize, dpi=100)
        plt.plot(x, y, lw=3, color='black', label='analytical')
        plt.plot(xmpas, ympas, marker='o', label='MPAS-O forward')
        plt.legend(frameon=False)
        plt.ylabel('Tidal amplitude (m)')
        plt.xlabel('Time (hrs)')
        plt.suptitle('Tidal amplitude forcing (right side) for MPAS-O and analytical')
        plt.savefig('ssh_t.png', bbox_inches='tight', dpi=200)
        plt.close()

        # plot cross-sections
        times = ['0.50', '0.05', '0.40', '0.15', '0.30', '0.25']
        fig, ax = plt.subplots(nrows=1,ncols=1, sharex=True, sharey=True)
        fig.text(0.04, 0.5, 'Channel depth (m)', va='center', rotation='vertical')
        fig.text(0.5, 0.02, 'Along channel distance (km)', ha='center')

        plt.xlim(0,25)
        plt.ylim(-1, 11)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        x = numpy.linspace(0,25,100)
        y = 10.0/25.0*x
        plt.plot(x, y, 'k-', lw=3)

        for atime in times:
            # factor of 1e-16 needed to account for annoying round-off issue to get right time slices
            plottime = int((float(atime)/0.2 + 1e-16)*24.0)
            #print('{} {} {}'.format(atime, plottime, ds.isel(Time=plottime).xtime.values))
            print('{} {}'.format(atime, plottime))
            ds = ds.drop_vars(numpy.setdiff1d([i for i in ds.variables], ['yCell','ssh']))
            ymean = ds.isel(Time=plottime).groupby('yCell').mean(dim=xarray.ALL_DIMS)
            x = ymean.yCell.values/1000.0
            y = ymean.ssh.values
            #print('ymin={} ymax={}\n{}\n{}'.format(y.min(), y.max(),x, y))
            print('ymin={} ymax={}\n'.format(y.min(), y.max()))
            plt.plot(x, -y)#, *args, **kwargs)
        plt.savefig('ssh_section_t.png', bbox_inches='tight', dpi=200)
        plt.close()

        ds.close()

