import numpy as np
import xarray
import matplotlib.pyplot as plt
import cmocean

from compass.step import Step
from compass.ocean.rpe import compute_rpe


class Viz(Step):
    """
    A step for plotting the results of a series of TODO runs in the merry-go-
    round test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    """
    def __init__(self, test_case, resolution):
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
        super().__init__(test_case=test_case, name='viz')
        self.resolution = resolution

        self.add_input_file(
            filename='initial_state.nc',
            target='../initial_state/init.nc')

        self.add_output_file(
            filename='section_{}.png'.format(resolution))

    def run(self):
        """
        Run this step of the test case
        """
        section = self.config['merry_go_round']
        nx = section.getint('nx')
        ny = section.getint('ny')
        print(self.outputs[0])
        _plot(nx, ny, self.outputs[0])


def _plot(nx, ny, filename):
    """
    Plot section of the merry-go-round TODO

    Parameters
    ----------
    nx : int
        The number of cells in the x direction

    ny : int
        The number of cells in the y direction (before culling)

    filename : str
        The output file name

    """

    # Note: ny does not currently get used
    plt.switch_backend('Agg')
    fig = plt.gcf()
    fig.set_size_inches(8.0,10.0)
    
    ds = xarray.open_dataset('../forward/output.nc')
    var1 = ds.velocityX
    var2 = ds.vertVelocityTop
    var3 = ds.tracer1
    nVertLevels = ds['nVertLevels'].size
    var11 = var1[:, 0:nx, :].values
    var22 = var2[:, 0:nx, :].values
    var33 = var3[:, 0:nx, :].values
    for iCell in range(nx):
        sumVert = 0.0
        for k in range(nVertLevels):
            sumVert = sumVert + var11[1, iCell, k]
    for k in range(nVertLevels):
        sumVert = 0.0
        for iCell in range(nx):
            sumVert = sumVert + var22[1, iCell, k]
    
    plt.title('merry-go-round')
    plt.subplot(2, 2, 1) 
    ax = plt.imshow(var11[1, :, :].T)
    plt.jet()
    plt.xlabel('x, # of cells')
    plt.ylabel('z, # of levels')
    plt.colorbar(ax, shrink=0.5)
    plt.title('horizontal velocity')
    
    plt.subplot(2, 2, 2)
    ax = plt.imshow(var22[1, :, :].T)
    plt.jet()
    plt.xlabel('x, # of cells')
    plt.colorbar(ax, shrink=0.5)
    plt.title('vertical velocity')
    
    plt.subplot(2, 2, 3)
    ax = plt.imshow(var33[0, :, :].T)
    plt.jet()
    plt.xlabel('x, # of cells')
    plt.ylabel('z, # of levels')
    plt.colorbar(ax, shrink=0.5)
    plt.title('tracer1 at t=0')
    
    plt.subplot(2, 2, 4)
    ax = plt.imshow(var33[1, :, :].T)
    plt.jet()
    plt.xlabel('x, # of cells')
    plt.colorbar(ax, shrink=0.5)
    plt.title('tracer1 at 6h')
    
    plt.savefig(filename)
