import numpy
import xarray
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step


class Viz(Step):
    """
    A step for plotting the results of the merry-go-round test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    """
    def __init__(self, test_case, resolution, name='viz'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        name: str
            The name of the step

        """
        super().__init__(test_case=test_case, name=name)
        self.resolution = resolution

        self.add_input_file(
            filename='initial_state.nc',
            target=f'../initial_state_{resolution}/init.nc')

        self.add_output_file(filename='section.png')

    def run(self):
        """
        Run this step of the test case
        """
        _plot(self.outputs[0], self.resolution)


def _plot(filename, resolution):
    """
    Plot section of the merry-go-round test case properties

    Parameters
    ----------
    filename : str
        The output file name

    resolution : str
        The resolution of the test case
    """

    plt.switch_backend('Agg')
    fig = plt.gcf()
    fig.set_size_inches(8.0, 10.0)

    ds = xarray.open_dataset(f'../forward_{resolution}/output.nc')
    var1 = ds.velocityX
    var2 = ds.vertVelocityTop
    var3 = ds.tracer1

    nVertLevels = ds['nVertLevels'].size
    nCells = ds['nCells'].size
    nx = int(nCells/4)

    xCell = ds.xCell.values
    zMid = ds.refZMid.values
    dx = xCell[1] - xCell[0]
    dz = zMid[0] - zMid[1]
    xCell = xCell + dx/2.
    x = numpy.insert(xCell[0:nx], [0], xCell[0] - dx/2.)
    zMid = zMid - dz/2.
    z = numpy.insert(zMid, [0], zMid[0] + dz/2.)
    X, Z = numpy.meshgrid(x, z)
    var11 = var1[:, 0:nx, :].values
    var22 = var2[:, 0:nx, :].values
    var33 = var3[:, 0:nx, :].values

    plt.subplot(2, 2, 1)
    plt.suptitle(f'Merry-go-round, horizontal resolution = {resolution}')

    ax = plt.pcolormesh(X, Z, var11[1, :, :].T)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.set_cmap('cmo.balance')
    plt.clim([-0.008, 0.008])
    plt.ylabel('z (m)')
    plt.colorbar(ax, shrink=0.5)
    plt.title('horizontal velocity')

    plt.subplot(2, 2, 2)
    ax = plt.pcolormesh(X, Z, var22[1, :, :-1].T)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.set_cmap('cmo.balance')
    plt.clim([-0.02, 0.02])
    plt.colorbar(ax, shrink=0.5)
    plt.title('vertical velocity')

    plt.subplot(2, 2, 3)
    ax = plt.pcolormesh(X, Z, var33[1, :, :].T)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.set_cmap('cmo.thermal')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.colorbar(ax, shrink=0.5)
    plt.title('tracer1 at t=0')

    plt.subplot(2, 2, 4)
    ax = plt.pcolormesh(X, Z,
                        numpy.subtract(var33[1, :, :].T,
                                       var33[0, :, :].T))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.set_cmap('cmo.curl')
    plt.clim([-0.05, 0.05])
    plt.xlabel('x (m)')
    plt.colorbar(ax, shrink=0.5)
    plt.title('delta(tracer1) at 6h')

    plt.savefig(filename)
