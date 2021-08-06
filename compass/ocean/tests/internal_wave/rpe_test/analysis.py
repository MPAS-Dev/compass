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

    fig = plt.gcf()
    nRow = 5
    nCol = 2
    iTime = [0, 1]
    time = ['1', '21']
    
    fig, axs = plt.subplots(nRow, nCol, figsize=(
        4.0 * nCol, 3.7 * nRow), constrained_layout=True)
    
    for iRow in range(nRow):
        ncfile = Dataset('output_' + str(iRow + 1) + '.nc', 'r')
        var = ncfile.variables['temperature']
        xtime = ncfile.variables['xtime']
        for iCol in range(nCol):
            ax = axs[iRow, iCol]
            dis = ax.imshow(
                var[iTime[iCol], 0::4, :].T, 
                extent=[0, 250, 500, 0], 
                aspect='0.5', 
                cmap='jet', 
                vmin=10, 
                vmax=20)
            if iRow == nRow - 1:
                ax.set_xlabel('x, km')
            if iCol == 0:
                ax.set_ylabel('depth, m')
            if iCol == nCol - 1:
                fig.colorbar(dis, ax=axs[iRow, iCol], aspect=10)
            ax.set_title("day {}, $\\nu_h=${}".format(time[iCol], nus[iRow]))
        ncfile.close()

    plt.savefig(filename)
