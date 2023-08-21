import matplotlib.pyplot as plt
import numpy
import pandas as pd
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
        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        self.add_output_file('vert_grid.png')
        self.add_output_file('velocity_max_t.png')

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

        y = ds.yCell.values / 1.e3
        plt.figure(figsize=[12, 6], dpi=100)
        plt.plot(y, zMid.values, 'b', label='zMid')
        plt.plot(y, zInterface.values, 'k', label='zTop')
        plt.plot(y, ds.ssh.values, '--g', label='SSH')
        plt.plot(y, -ds.bottomDepth.values, '--r', label='zBed')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (m)')
        plt.legend()
        plt.savefig('vert_grid.png', dpi=200)
        plt.close()
        ds.close()

        ds = xarray.open_dataset('output.nc')
        # Plot the time series of max velocity
        plt.figure(figsize=[12, 6], dpi=100)
        umax = numpy.amax(ds.velocityX[:, :, 0].values, axis=1)
        vmax = numpy.amax(ds.velocityY[:, :, 0].values, axis=1)
        t = ds.daysSinceStartOfSim.values
        time = pd.to_timedelta(t) / 1.e9
        plt.plot(time, umax, 'k', label='u')
        plt.plot(time, vmax, 'b', label='v')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.savefig('velocity_max_t.png', dpi=200)
        plt.close()
