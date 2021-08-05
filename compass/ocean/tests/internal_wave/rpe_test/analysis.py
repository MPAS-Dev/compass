import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step


class Analysis(Step):
    """
    A step for plotting the results of a series of RPE runs in the internal
    wave test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    nus : list of float
        A list of viscosities
    """
    def __init__(self, test_case, nus):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        nus : list of float
            A list of viscosities
        """
        super().__init__(test_case=test_case, name='analysis')
        self.nus = nus

        for index, nu in enumerate(nus):
            self.add_input_file(
                filename='output_{}.nc'.format(index+1),
                target='../rpe_test_{}_nu_{}/output.nc'.format(index+1, nu))

        self.add_output_file(
            filename='sections_internal_wave.png')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['internal_wave']
        nx = section.getint('nx')
        ny = section.getint('ny')
        _plot(nx, ny, self.outputs[0], self.nus)


def _plot(nx, ny, filename, nus):
    """
    Plot section of the internal wave at different viscosities

    Parameters
    ----------
    nx : int
        The number of cells in the x direction

    ny : int
        The number of cells in the y direction (before culling)

    filename : str
        The output file name

    nus : list of float
        The viscosity values
    """

    plt.switch_backend('Agg')

    nRow = 1
    nCol = 5
    iTime = [0]
    time = ['20']

    fig, axs = plt.subplots(nRow, nCol, figsize=(
        2.1 * nCol, 5.0 * nRow), constrained_layout=True)

    for iCol in range(nCol):
        for iRow in range(nRow):
            ncfile = Dataset('output_{}.nc'.format(iCol + 1), 'r')
            var = ncfile.variables['temperature']
            var1 = np.reshape(var[iTime[iRow], :, 0], [ny, nx])
            # flip in y-dir
            var = np.flipud(var1)

            # Every other row in y needs to average two neighbors in x on
            # planar hex mesh
            var_avg = var
            for j in range(0, ny, 2):
                for i in range(1, nx - 2):
                    var_avg[j, i] = (var[j, i + 1] + var[j, i]) / 2.0

            if nRow == 1:
                ax = axs[iCol]
            else:
                ax = axs[iRow, iCol]
            dis = ax.imshow(
                var_avg,
                extent=[0, 160, 0, 500],
                cmap='cmo.thermal',
                vmin=11.8,
                vmax=13.0)
            ax.set_title("day {}, $\\nu_h=${}".format(time[iRow], nus[iCol]))
            ax.set_xticks(np.arange(0, 161, step=40))
            ax.set_yticks(np.arange(0, 501, step=50))

            if iRow == nRow - 1:
                ax.set_xlabel('x, km')
            if iCol == 0:
                ax.set_ylabel('y, km')
            if iCol == nCol - 1:
                if nRow == 1:
                    fig.colorbar(dis, ax=axs[nCol - 1], aspect=40)
                else:
                    fig.colorbar(dis, ax=axs[iRow, nCol - 1], aspect=40)
            ncfile.close()

    plt.savefig(filename)
