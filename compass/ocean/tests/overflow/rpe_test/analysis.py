import numpy as np
import xarray
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step
from compass.ocean.rpe import compute_rpe


class Analysis(Step):
    """
    A step for plotting the results of a series of RPE runs in the overflow
    test group

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
            filename='sections_overflow_{}.png'.format(resolution))
        self.add_output_file(filename='rpe_t.png')

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['overflow']
        nx = section.getint('nx')
        ny = section.getint('ny') - 2
        rpe = compute_rpe()
        _plot(nx, ny, self.outputs[0], self.nus, rpe)


def _plot(nx, ny, filename, nus, rpe):
    """
    TODO change section from nx vs ny to ny vs. nz
    Plot section of the overflow at different viscosities

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
    time = 6/24

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

    # prep mesh quantities
    ds = xarray.open_dataset('../initial_state/ocean.nc')
    ds = ds.sortby('yEdge')
    xEdge = ds.xEdge
    yEdge = ds.yEdge
    cellsOnEdge = ds.cellsOnEdge
    nVertLevels = ds.sizes['nVertLevels']

    xEdge_mid = np.median(xEdge)
    edgeMask_x = np.equal(xEdge, xEdge_mid)
    yEdge_x = yEdge[edgeMask_x]
    cellsOnEdge_x = cellsOnEdge[edgeMask_x, :]
    cell1Index = np.subtract(cellsOnEdge_x[:, 0], 1)
    cell2Index = np.subtract(cellsOnEdge_x[:, 1], 1)
    nEdges_x = len(yEdge_x)

    fig, axs = plt.subplots(num_files, 1, figsize=(
        5.0, 2.1 * num_files), constrained_layout=True)
    fig.suptitle(f'Temperature, Overflow test case')

    for iCol in range(num_files):

        ax = axs[iCol]
        ds = xarray.open_dataset('output_{}.nc'.format(iCol + 1))

        # Get the output times again
        # Don't assume that the output times are the same for all files
        times = ds.daysSinceStartOfSim.values
        times = np.divide(times.tolist(), nanosecondsPerDay)
        tidx = np.argmin(np.abs(times-time))
        ds = ds.isel(Time=tidx)

        ds = ds.sortby('yEdge')

        # Compute the layer interfaces depths across the cross-section
        layerThickness = ds.layerThickness
        layerThickness_cell1 = layerThickness[cell1Index, :]
        layerThickness_cell2 = layerThickness[cell2Index, :]
        layerThickness_x = layerThickness_cell2[1:, :]

        ssh = ds.ssh
        ssh_cell1 = ssh[cell1Index]
        ssh_cell2 = ssh[cell2Index]
        ssh_x = ssh_cell2[1:]

        zIndex = xarray.DataArray(data=np.arange(nVertLevels),
                                  dims='nVertLevels')

        zEdgeInterface = np.zeros((nEdges_x, nVertLevels + 1))
        zEdgeInterface[:, 0] = 0.5 * (ssh_cell1.values + ssh_cell2.values)
        for zIndex in range(nVertLevels):
            thickness1 = layerThickness_cell1.isel(nVertLevels=zIndex)
            thickness1 = thickness1.fillna(0.)
            thickness2 = layerThickness_cell2.isel(nVertLevels=zIndex)
            thickness2 = thickness2.fillna(0.)
            zEdgeInterface[:, zIndex + 1] = \
                zEdgeInterface[:, zIndex] - \
                0.5 * (thickness1.values + thickness2.values)

        _, yEdges_mesh = np.meshgrid(zEdgeInterface[0, :], yEdge_x)

        # Retrieve the temperature field
        temperature = ds.temperature
        temperature_x = temperature[cell2Index[1:], :]

        # Plot
        dis = ax.pcolormesh(np.divide(yEdges_mesh, 1e3),
                            zEdgeInterface,
                            temperature_x.values, cmap='viridis',
                            vmin=10, vmax=20)
        ax.set_title(f'hour {int(times[tidx]*24.)}, '
                     f'$\\nu_h=${nus[iCol]}')

        ax.set_ylabel('z (m)')
        ax.set_xlim([10, 100])
        if iCol == 0:
            fig.colorbar(dis)
        if iCol == num_files - 1:
            ax.set_xlabel('y (km)')

    plt.savefig(filename)
