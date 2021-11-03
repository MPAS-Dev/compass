import numpy as np
import xarray
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step
from compass.ocean.rpe import compute_rpe


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

        self.add_input_file(
            filename='initial_state.nc',
            target='../initial_state/ocean.nc')

        for index, nu in enumerate(nus):
            self.add_input_file(
                filename='output_{}.nc'.format(index+1),
                target='../rpe_test_{}_nu_{}/output.nc'.format(index+1, nu))

        self.add_output_file(
            filename='sections_internal_wave.png')
        self.add_output_file(filename='rpe_t.png')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['internal_wave']
        nx = section.getint('nx')
        ny = section.getint('ny')
        rpe = compute_rpe(num_files=len(self.nus))
        _plot(nx, ny, self.outputs[0], self.nus, rpe)


def _plot(nx, ny, filename, nus, rpe):
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

    rpe : float, dim len(nu) x len(time)
    """

    plt.switch_backend('Agg')
    nanosecondsPerDay = 8.64e13
    num_files = len(nus)
    time = [1, 21]

    ds = xarray.open_dataset('output_1.nc')
    times = ds.daysSinceStartOfSim.values
    times = times.tolist()
    times = np.divide(times, nanosecondsPerDay)

    fig = plt.figure()
    for i in range(num_files):
        rpe_norm = np.divide((rpe[i, :]-rpe[i, 0]), rpe[i, 0])
        plt.plot(times, rpe_norm,
                 label="$\\nu_h=${}".format(nus[i]))
    plt.xlabel('Time, days')
    plt.ylabel('RPE-RPE(0)/RPE(0)')
    plt.legend()
    plt.savefig('rpe_t.png')
    plt.close(fig)

    nCol = len(time)
    nRow = len(nus)

    fig, axs = plt.subplots(nRow, nCol, figsize=(
        4.0 * nCol, 3.7 * nRow), constrained_layout=True)

    for iRow in range(nRow):
        ds = xarray.open_dataset('output_{}.nc'.format(iRow + 1))
        times = ds.daysSinceStartOfSim.values
        times = np.divide(times.tolist(), nanosecondsPerDay)
        var = ds.temperature.values
        for iCol in range(nCol):
            ax = axs[iRow, iCol]
            tidx = np.argmin(np.abs(times-time[iCol]))
            dis = ax.imshow(
                var[tidx, 0::4, :].T,
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

    plt.savefig('sections_internal_wave.png')
