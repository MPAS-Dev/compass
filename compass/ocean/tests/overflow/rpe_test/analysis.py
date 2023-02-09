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
                filename=f'output_{index+1}.nc',
                target=f'../rpe_test_{index+1}_nu_{int(nu)}/output.nc')

        self.add_output_file(
            filename=f'sections_overflow_{resolution}.png')
        self.add_output_file(filename='rpe_t.png')

    def run(self):
        """
        Run this step of the test case
        """
        rpe = compute_rpe()
        _plot(self.outputs[0], self.nus, rpe)


def _plot(filename, nus, rpe):
    """
    Plot section of the overflow at different viscosities

    Parameters
    ----------

    filename : str
        The output file name

    nus : list of float
        The viscosity values

    rpe : numpy.ndarray
        The reference potential energy with size len(nu) x len(time)
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
                 label=f"$\\nu_h=${nus[i]}")
    plt.xlabel('Time, days')
    plt.ylabel('RPE-RPE(0)/RPE(0)')
    plt.legend()
    plt.savefig('rpe_t.png')
    plt.close(fig)

    # prep mesh quantities
    ds = xarray.open_dataset('../initial_state/ocean.nc')
    ds = ds.sortby('yEdge')
    x_edge = ds.xEdge
    y_edge = ds.yEdge
    cells_on_edge = ds.cellsOnEdge
    n_vert_levels = ds.sizes['nVertLevels']

    x_edge_mid = np.median(x_edge)
    edge_mask_x = np.equal(x_edge, x_edge_mid)
    y_edge_x = y_edge[edge_mask_x]
    cells_on_edge_x = cells_on_edge[edge_mask_x, :]
    cell1_index = np.subtract(cells_on_edge_x[:, 0], 1)
    cell2_index = np.subtract(cells_on_edge_x[:, 1], 1)
    nEdges_x = len(y_edge_x)

    fig, axs = plt.subplots(num_files, 1, figsize=(
        5.0, 2.1 * num_files), constrained_layout=True)
    fig.suptitle('Temperature, Overflow test case')

    for iCol in range(num_files):

        ax = axs[iCol]
        ds = xarray.open_dataset(f'output_{iCol + 1}.nc')

        # Get the output times again
        # Don't assume that the output times are the same for all files
        times = ds.daysSinceStartOfSim.values
        times = np.divide(times.tolist(), nanosecondsPerDay)
        t_idx = np.argmin(np.abs(times-time))
        ds = ds.isel(Time=t_idx)

        ds = ds.sortby('yEdge')

        # Compute the layer interfaces depths across the cross-section
        layer_thickness = ds.layerThickness
        layer_thickness_cell1 = layer_thickness[cell1_index, :]
        layer_thickness_cell2 = layer_thickness[cell2_index, :]

        ssh = ds.ssh
        ssh_cell1 = ssh[cell1_index]
        ssh_cell2 = ssh[cell2_index]

        z_interface = np.zeros((nEdges_x, n_vert_levels + 1))
        z_interface[:, 0] = 0.5 * (ssh_cell1.values + ssh_cell2.values)
        for z_index in range(n_vert_levels):
            thickness1 = layer_thickness_cell1.isel(nVertLevels=z_index)
            thickness1 = thickness1.fillna(0.)
            thickness2 = layer_thickness_cell2.isel(nVertLevels=z_index)
            thickness2 = thickness2.fillna(0.)
            z_interface[:, z_index + 1] = \
                z_interface[:, z_index] - \
                0.5 * (thickness1.values + thickness2.values)

        _, y_edges_mesh = np.meshgrid(z_interface[0, :], y_edge_x)

        # Retrieve the temperature field
        temperature = ds.temperature
        temperature_x = temperature[cell2_index[1:], :]

        # Plot
        p = ax.pcolormesh(np.divide(y_edges_mesh, 1e3),
                          z_interface,
                          temperature_x.values,
                          cmap='viridis', vmin=10, vmax=20)
        ax.set_title(f'hour {int(times[t_idx]*24.)}, '
                     f'$\\nu_h=${nus[iCol]}')

        ax.set_ylabel('z (m)')
        ax.set_xlim([10, 100])
        if iCol == 0:
            fig.colorbar(p)
        if iCol == num_files - 1:
            ax.set_xlabel('y (km)')

    plt.savefig(filename)
