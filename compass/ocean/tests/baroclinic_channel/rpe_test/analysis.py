import numpy as np
import xarray
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step
from compass.ocean.rpe import compute_rpe


class Analysis(Step):
    """
    A step for plotting the results of a series of RPE runs in the baroclinic
    channel test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    nus : list of float
        A list of viscosities
    """
    def __init__(self, test_case, resolution, nus):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        nus : list of float
            A list of viscosities
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolution = resolution
        self.nus = nus

        self.add_input_file(
            filename='initial_state.nc',
            target='../initial_state/ocean.nc')

        for index, nu in enumerate(nus):
            self.add_input_file(
                filename='output_{}.nc'.format(index+1),
                target='../rpe_test_{}_nu_{}/output.nc'.format(index+1, nu))

        self.add_output_file(
            filename='sections_baroclinic_channel_{}.png'.format(resolution))
        self.add_output_file(filename='rpe_t.png')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['baroclinic_channel']
        nx = section.getint('nx')
        ny = section.getint('ny')
        rpe = compute_rpe()
        _plot(nx, ny, self.outputs[0], self.nus, rpe)


def _plot(nx, ny, filename, nus, rpe):
    """
    Plot section of the baroclinic channel at different viscosities

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
    time = 20

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

    fig, axs = plt.subplots(1, num_files, figsize=(
        2.1 * num_files, 5.0), constrained_layout=True)

    for iCol in range(num_files):
        ds = xarray.open_dataset('output_{}.nc'.format(iCol + 1))
        times = ds.daysSinceStartOfSim.values
        times = np.divide(times.tolist(), nanosecondsPerDay)
        tidx = np.argmin(np.abs(times-time))
        var = ds.temperature.values
        var1 = np.reshape(var[tidx, :, 0], [ny, nx])
        # flip in y-dir
        var = np.flipud(var1)

        # Every other row in y needs to average two neighbors in x on
        # planar hex mesh
        var_avg = var
        for j in range(0, ny, 2):
            for i in range(1, nx - 2):
                var_avg[j, i] = (var[j, i + 1] + var[j, i]) / 2.0

        ax = axs[iCol]
        dis = ax.imshow(
            var_avg,
            extent=[0, 160, 0, 500],
            cmap='cmo.thermal',
            vmin=11.8,
            vmax=13.0)
        ax.set_title("day {}, $\\nu_h=${}".format(
            int(times[tidx]), nus[iCol]))
        ax.set_xticks(np.arange(0, 161, step=40))
        ax.set_yticks(np.arange(0, 501, step=50))

        ax.set_xlabel('x, km')
        if iCol == 0:
            ax.set_ylabel('y, km')
        if iCol == num_files - 1:
            fig.colorbar(dis, ax=axs[num_files - 1], aspect=40)

    plt.savefig(filename)
