import matplotlib.pyplot as plt
import numpy
import xarray

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing a cross-section through the internal wave
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

        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        self.add_output_file('uNormal_depth_section_t0.png')
        self.add_output_file('pt_depth_section_t0.png')
        self.add_output_file('sa_depth_section_t0.png')
        self.add_output_file('layerThickness_depth_section_t0.png')

    def run(self):
        """
        Run this step of the test case
        """

        ds = xarray.open_dataset('output.nc')
        figsize = [6.4, 4.8]
        if 'Time' not in ds.dims:
            print('Dataset missing time dimension')
            return
        nSteps = ds.sizes['Time']  # number of timesteps
        tend = nSteps - 1

        for j in [0, tend]:
            ds1 = ds.isel(Time=j)

            # prep all variables for uNormal plot
            ds1 = ds1.sortby('yEdge')

            nCells = ds1.sizes['nCells']
            nEdges = ds1.sizes['nEdges']
            nVertLevels = ds1.sizes['nVertLevels']

            xEdge = numpy.zeros((nEdges))
            xEdge = ds1.xEdge
            yCell = numpy.zeros((nCells))
            yCell = ds1.yCell

            xEdge_mid = numpy.median(xEdge)
            edgeMask_x = numpy.equal(xEdge, xEdge_mid)

            zIndex = xarray.DataArray(data=numpy.arange(nVertLevels),
                                      dims='nVertLevels')

            zInterface = numpy.zeros((nCells, nVertLevels + 1))
            zInterface[:, 0] = ds1.ssh.values
            for zIndex in range(nVertLevels):
                thickness = ds1.layerThickness.isel(nVertLevels=zIndex)
                thickness = thickness.fillna(0.)
                zInterface[:, zIndex + 1] = \
                    zInterface[:, zIndex] - thickness.values

            zMid = numpy.zeros((nCells, nVertLevels))
            for zIndex in range(nVertLevels):
                zMid[:, zIndex] = (zInterface[:, zIndex] +
                                   numpy.divide(zInterface[:, zIndex + 1] -
                                                zInterface[:, zIndex], 2.))

            # Solve for lateral boundaries of uNormal at cell centers for
            # x-section
            cellsOnEdge = ds1.cellsOnEdge
            cellsOnEdge_x = cellsOnEdge[edgeMask_x, :]
            yEdges = numpy.zeros((len(cellsOnEdge_x) + 1))
            for i in range(len(cellsOnEdge_x)):
                if cellsOnEdge[i, 1] == 0:
                    yEdges[i] = yCell[cellsOnEdge_x[i, 0] - 1]
                    yEdges[i + 1] = yCell[cellsOnEdge_x[i, 0] - 1]
                elif cellsOnEdge[i, 1] == 0:
                    yEdges[i] = yCell[cellsOnEdge_x[i, 1] - 1]
                    yEdges[i + 1] = yCell[cellsOnEdge_x[i, 1] - 1]
                else:
                    yEdges[i] = min(yCell[cellsOnEdge_x[i, 0] - 1],
                                    yCell[cellsOnEdge_x[i, 1] - 1])
                    yEdges[i + 1] = max(yCell[cellsOnEdge_x[i, 0] - 1],
                                        yCell[cellsOnEdge_x[i, 1] - 1])

            zInterfaces_mesh, yEdges_mesh = numpy.meshgrid(zInterface[0, :],
                                                           yEdges)

            normalVelocity = numpy.zeros((nCells, nVertLevels))
            normalVelocity = ds1.normalVelocity
            normalVelocity_xmesh = normalVelocity[edgeMask_x, :]

            # Figures
            plt.figure(figsize=figsize, dpi=100)
            cmax = numpy.max(numpy.abs(normalVelocity_xmesh.values))
            plt.pcolormesh(numpy.divide(yEdges_mesh, 1e3),
                           zInterfaces_mesh,
                           normalVelocity_xmesh.values,
                           cmap='RdBu', vmin=-1. * cmax, vmax=cmax)
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('uNormal (m/s)')
            plt.savefig('uNormal_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            # ------------------------------------------------------------------
            # Plot cell-centered variables
            # ------------------------------------------------------------------
            # Prep variables for cell quantities
            cellIndex = numpy.subtract(cellsOnEdge_x[1:, 0], 1)
            yEdge = numpy.zeros((nEdges))
            yEdge = ds1.yEdge
            yEdge_x = yEdge[edgeMask_x]

            zInterfaces_mesh, yCells_mesh = numpy.meshgrid(zInterface[0, :],
                                                           yEdge_x)

            # Import cell quantities
            layerThickness = numpy.zeros((nCells, nVertLevels))
            layerThickness = ds1.layerThickness
            layerThickness_x = layerThickness[cellIndex, :]
            temperature = numpy.zeros((nCells, nVertLevels))
            temperature = ds1.temperature
            temperature_z = temperature.mean(dim='nCells')
            zMid_z = zMid.mean(axis=0)
            temperature_x = temperature[cellIndex, :]
            salinity = numpy.zeros((nCells, nVertLevels))
            salinity = ds1.salinity
            salinity_x = salinity[cellIndex, :]
            w = numpy.zeros((nCells, nVertLevels))
            w = ds1.vertVelocityTop
            w_x = w[cellIndex, :]

            # Figures
            plt.figure(figsize=figsize, dpi=100)
            plt.plot(temperature_z.values, zMid_z)
            plt.xlabel('PT (C)')
            plt.ylabel('z (m)')
            plt.savefig('pt_depth_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_mesh,
                           temperature_x.values, cmap='viridis')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('PT (C)')
            plt.savefig('pt_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_mesh,
                           salinity_x.values, cmap='viridis')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('SA (g/kg)')
            plt.savefig('sa_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_mesh,
                           w_x.values[:, :-1], cmap='viridis')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('h (m)')
            plt.savefig('w_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_mesh,
                           layerThickness_x.values, cmap='viridis')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('h (m)')
            plt.savefig('layerThickness_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()
