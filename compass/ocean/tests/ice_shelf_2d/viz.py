import matplotlib.pyplot as plt
import numpy
import xarray

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing a cross-section through the 2D ice-shelf domain
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

        self.add_input_file(filename='initial_state.nc',
                            target='../initial_state/initial_state.nc')
        self.add_output_file('vert_grid.png')

    def run(self):
        """
        Run this step of the test case
        """

        ds = xarray.open_dataset('initial_state.nc')

        if 'Time' in ds.dims:
            ds = ds.isel(Time=0)

        ds = ds.groupby('yCell').mean(dim='nCells')

        nCells = ds.sizes['yCell']
        nVertLevels = ds.sizes['nVertLevels']

        zIndex = xarray.DataArray(data=numpy.arange(nVertLevels),
                                  dims='nVertLevels')

        minLevelCell = ds.minLevelCell - 1
        maxLevelCell = ds.maxLevelCell - 1

        cellMask = numpy.logical_and(zIndex >= minLevelCell,
                                     zIndex <= maxLevelCell)

        zIndex = xarray.DataArray(data=numpy.arange(nVertLevels + 1),
                                  dims='nVertLevelsP1')

        interfaceMask = numpy.logical_and(zIndex >= minLevelCell,
                                          zIndex <= maxLevelCell + 1)

        zMid = ds.zMid.where(cellMask)

        zInterface = numpy.zeros((nCells, nVertLevels + 1))
        zInterface[:, 0] = ds.ssh.values
        for zIndex in range(nVertLevels):
            thickness = ds.layerThickness.isel(nVertLevels=zIndex)
            thickness = thickness.fillna(0.)
            zInterface[:, zIndex + 1] = \
                zInterface[:, zIndex] - thickness.values

        zInterface = xarray.DataArray(data=zInterface,
                                      dims=('yCell', 'nVertLevelsP1'))
        zInterface = zInterface.where(interfaceMask)

        plt.figure(figsize=[24, 12], dpi=100)
        plt.plot(ds.yCell.values, zMid.values, 'b')
        plt.plot(ds.yCell.values, zInterface.values, 'k')
        plt.plot(ds.yCell.values, ds.ssh.values, 'g')
        plt.plot(ds.yCell.values, -ds.bottomDepth.values, 'g')
        plt.savefig('vert_grid.png', dpi=200)
        plt.close()
