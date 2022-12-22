import xarray
import numpy
import matplotlib.pyplot as plt

from compass.step import Step
import cmocean


class Viz(Step):
    """
    A step for visualizing a cross-section through the domain
    """
    def __init__(self, test_case, resolution, forcing, do_comparison=False):
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

        if resolution == '1m':
            suffix = 'g128_l128'

        if do_comparison:
            if forcing == 'cooling':
                forcing_name = 'c02'
            elif forcing == 'evaporation':
                forcing_name = 'e04'
            else:
                #TODO change to log
                print('Comparison simulation not available for this configuration')
                do_comparison = False

        if do_comparison:
            filename = f'case_{forcing_name}_{suffix}.nc'
            print(f'Compare to {filename}')
            self.add_input_file(filename='palm.nc', target=filename,
                                database='turbulence_closure')

        self.do_comparison = do_comparison

    def run(self):
        """
        Run this step of the test case
        """

        ds = xarray.open_dataset('output.nc')
        ds = ds.sortby('yEdge')

        if self.do_comparison:
            dsPalm = xarray.open_dataset('palm.nc')
        dsInit = xarray.open_dataset('../forward/init.nc')
        ds0 = ds.isel(Time=0)
        figsize = [6.4, 4.8]
        markersize = 20
        if 'Time' not in ds.dims:
            print('Dataset missing time dimension')
            return
        nSteps = ds.sizes['Time']  # number of timesteps
        tend = nSteps - 1

        nCells = dsInit.sizes['nCells']
        nEdges = dsInit.sizes['nEdges']
        nVertLevels = dsInit.sizes['nVertLevels']

        xEdge = dsInit.xEdge
        yEdge = dsInit.yEdge
        xCell = dsInit.xCell
        yCell = dsInit.yCell

        xEdge_mid = numpy.median(xEdge)
        edgeMask_x = numpy.equal(xEdge, xEdge_mid)

        # Solve for lateral boundaries of uNormal at cell centers for
        # x-section
        cellsOnEdge = dsInit.cellsOnEdge
        cellsOnEdge_x = cellsOnEdge[edgeMask_x, :]
        yEdges = numpy.zeros((len(cellsOnEdge_x)+1))
        for i in range(len(cellsOnEdge_x)):
            if cellsOnEdge[i, 1] == 0:
                yEdges[i] = yCell[cellsOnEdge_x[i, 0] - 1]
                yEdges[i+1] = yCell[cellsOnEdge_x[i, 0] - 1]
            elif cellsOnEdge[i, 1] == 0:
                yEdges[i] = yCell[cellsOnEdge_x[i, 1] - 1]
                yEdges[i+1] = yCell[cellsOnEdge_x[i, 1] - 1]
            else:
                yEdges[i] = min(yCell[cellsOnEdge_x[i, 0] - 1],
                                yCell[cellsOnEdge_x[i, 1] - 1])
                yEdges[i+1] = max(yCell[cellsOnEdge_x[i, 0] - 1],
                                  yCell[cellsOnEdge_x[i, 1] - 1])

        # Prep variables for cell quantities
        cellIndex = numpy.subtract(cellsOnEdge_x[1:, 0], 1)
        yEdge_x = yEdge[edgeMask_x]

        # prep all variables for uNormal plot
        zInterface = dsInit.refInterfaces
        zMid = dsInit.refZMid

        zInterfaces_edge_mesh, yEdges_mesh = numpy.meshgrid(zInterface,
                                                       yEdges)
        zInterfaces_cell_mesh, yCells_mesh = numpy.meshgrid(zInterface,
                                                       yEdge_x)
        temperature0 = ds0.temperature
        temperature0_z = temperature0.mean(dim='nCells')

        for j in [tend]:
            ds1 = ds.isel(Time=j)

            normalVelocity = ds1.normalVelocity
            normalVelocity_xmesh = normalVelocity[edgeMask_x, :]

            velocityZonal = ds1.velocityZonal
            velocityZonal_z = velocityZonal.mean(dim='nCells')
            velocityMeridional = ds1.velocityMeridional
            velocityMeridional_z = velocityMeridional.mean(dim='nCells')

            # Import cell quantities
            layerThickness = ds1.layerThickness
            layerThickness_x = layerThickness[cellIndex, :]
            temperature = ds1.temperature
            temperature_z = temperature.mean(dim='nCells')
            temperature_x = temperature[cellIndex, :]
            salinity = ds1.salinity
            salinity_z = salinity.mean(dim='nCells')
            salinity_x = salinity[cellIndex, :]
            w = ds1.vertVelocityTop
            w_x = w[cellIndex, :]
            w_zTop = w[:, 0]

            # Figures
            plt.figure(figsize=figsize, dpi=100)
            cmax = numpy.max(numpy.abs(normalVelocity_xmesh.values))
            plt.pcolormesh(numpy.divide(yEdges_mesh, 1e3),
                           zInterfaces_edge_mesh,
                           normalVelocity_xmesh.values,
                           cmap='cmo.balance', vmin=-1.*cmax, vmax=cmax)
            cbar = plt.colorbar()
            cbar.ax.set_title('uNormal (m/s)')
            plt.savefig('uNormal_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            # ------------------------------------------------------------------
            # Plot cell-centered variables
            # ------------------------------------------------------------------
            # Figures
            plt.figure(figsize=figsize, dpi=100)
            plt.plot(temperature_z.values, zMid)
            plt.xlabel('PT (C)')
            plt.ylabel('z (m)')
            plt.savefig('pt_depth_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_cell_mesh,
                           temperature_x.values, cmap='cmo.thermal')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('PT (C)')
            plt.savefig('pt_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.plot(numpy.subtract(temperature_z.values,
                                    temperature0_z.values),
                     zMid)
            plt.xlabel('delta PT (C)')
            plt.ylabel('z (m)')
            plt.savefig('dpt_depth_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.plot(salinity_z.values, zMid)
            plt.xlabel('SA (g/kg)')
            plt.ylabel('z (m)')
            plt.savefig('sa_depth_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_cell_mesh,
                           salinity_x.values, cmap='cmo.haline')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('SA (g/kg)')
            plt.savefig('sa_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.plot(velocityZonal_z.values, zMid)
            plt.plot(velocityMeridional_z.values, zMid, '--')
            plt.xlabel('u,v (m/s)')
            plt.ylabel('z (m)')
            plt.savefig('uv_depth_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            cmax = numpy.max(numpy.abs(w_x.values))
            plt.pcolormesh(numpy.divide(yCells_mesh[:-1,:], 1e3),
                           zInterfaces_cell_mesh[:-1,:],
                           w_x.values, cmap='cmo.balance',
                           vmin=-1.*cmax, vmax=cmax)
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('w (m/s)')
            plt.savefig('w_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            cmax = numpy.max(numpy.abs(w_zTop.values))
            plt.scatter(numpy.divide(xCell, 1e3),
                        numpy.divide(yCell, 1e3),
                        s=5, c=w_zTop.values,
                        cmap='cmo.balance', vmin=-1.*cmax, vmax=cmax)
            plt.xlabel('x (km)')
            plt.ylabel('y (km)')
            cbar = plt.colorbar()
            cbar.ax.set_title('w (m/s)')
            plt.savefig('w_top_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()

            plt.figure(figsize=figsize, dpi=100)
            plt.pcolormesh(numpy.divide(yCells_mesh, 1e3),
                           zInterfaces_cell_mesh,
                           layerThickness_x.values, cmap='viridis')
            plt.xlabel('y (km)')
            plt.ylabel('z (m)')
            cbar = plt.colorbar()
            cbar.ax.set_title('h (m)')
            plt.savefig('layerThickness_depth_section_t{}.png'.format(j),
                        bbox_inches='tight', dpi=200)
            plt.close()
