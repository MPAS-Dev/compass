import xarray
import numpy as np
import time
import math

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for stratified
    seiche test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
        """
        super().__init__(test_case=test_case, name='initial_state') 

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        timeStart = time.time()

        section = config['horizontal_grid']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=False)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        section = config['stratified_seiche'] 
        maxDepth = section.getfloat('maxDepth')
        nVertLevels = section.getint('nVertLevels')
        config_eos_linear_alpha = section.getfloat('eos_linear_alpha')
        config_eos_linear_beta = section.getfloat('eos_linear_beta')
        config_eos_linear_Tref = section.getfloat('eos_linear_Tref')
        config_eos_linear_Sref = section.getfloat('eos_linear_Sref')
        config_eos_linear_densityref = section.getfloat(
            'eos_linear_densityref')
        deltaRho = section.getfloat('deltaRho')
        interfaceThick = section.getfloat('interfaceThick')
        amplitude = section.getfloat('amplitude')
        wavenumber = section.getfloat('wavenumber')

        # comment('obtain dimensions and mesh variables')
        vertical_coordinate = 'uniform'

        ds = dsMesh.copy()
        nCells = ds.nCells.size
        nEdges = ds.nEdges.size
        nVertices = ds.nVertices.size

        xCell = ds.xCell
        yCell = ds.yCell
        xEdge = ds.xEdge
        yEdge = ds.yEdge
        angleEdge = ds.angleEdge
        cellsOnEdge = ds.cellsOnEdge
        edgesOnCell = ds.edgesOnCell 

        # Adjust coordinates so first edge is at zero in x and y
        xOffset = xEdge.min()
        xCell -= xOffset
        xEdge -= xOffset
        yOffset = np.min(yEdge)
        yCell -= yOffset
        yEdge -= yOffset

        # initialize velocity field 
        u = np.zeros([1, nEdges, nVertLevels])

        # comment('create and initialize variables')
        time1 = time.time()

        varsZ = ['refLayerThickness', 'refBottomDepth', 'refZMid',
            'vertCoordMovementWeights']
        for var in varsZ:
            globals()[var] = np.nan * np.ones(nVertLevels)

        vars2D = ['ssh', 'bottomDepth', 'surfaceStress',
            'atmosphericPressure', 'boundaryLayerDepth']
        for var in vars2D:
            globals()[var] = np.nan * np.ones(nCells)
        maxLevelCell = np.ones(nCells, dtype=np.int32)

        vars3D = [ 'layerThickness', 'temperature', 'salinity',
            'zMid', 'density']
        for var in vars3D:
            globals()[var] = np.nan * np.ones([1, nCells, nVertLevels])
        restingThickness = np.nan * np.ones([nCells, nVertLevels])

        # Note that this line shouldn't be required, but if layerThickness is
        # initialized with nans, the simulation dies. It must multiply by a
        # a nan on a land cell on an edge, and then multiply by zero.
        layerThickness[:] = -1e34

        # equally spaced layers
        refLayerThickness[:] = maxDepth / nVertLevels
        refBottomDepth[0] = refLayerThickness[0]
        refZMid[0] = -0.5 * refLayerThickness[0]
        for k in range(1, nVertLevels):
            refBottomDepth[k] = refBottomDepth[k - 1] + refLayerThickness[k]
            refZMid[k] = -refBottomDepth[k - 1] - 0.5 * refLayerThickness[k]

        # SSH
        ssh[:] = 0.0

        # Compute maxLevelCell and layerThickness for z-level
        # (variation only on top)
        if (vertical_coordinate == 'z'):
            vertCoordMovementWeights[:] = 0.0
            vertCoordMovementWeights[0] = 1.0
            for iCell in range(0, nCells):
                maxLevelCell[iCell] = nVertLevels - 1
                bottomDepth[iCell] = refBottomDepth[nVertLevels - 1] 
                layerThickness[0, iCell, :] = refLayerThickness[:]
                layerThickness[0, iCell, 0] += ssh[iCell]
            restingThickness[:, :] = layerThickness[0, :, :]
        # Compute maxLevelCell and layerThickness for uniform
        elif (vertical_coordinate == 'uniform'):
            vertCoordMovementWeights[:] = 1.0
            vertCoordMovementWeights[0] = 1.0
            for iCell in range(0, nCells):
                maxLevelCell[iCell] = nVertLevels - 1
                bottomDepth[iCell] = refBottomDepth[nVertLevels - 1]
                layerThickness[0, iCell, :] = refLayerThickness[:] + \
                    ssh[iCell]/nVertLevels
            restingThickness[:, :] = refLayerThickness[:]

        # Compute zMid (same, regardless of vertical coordinate)
        for iCell in range(0, nCells):
            k = maxLevelCell[iCell]
            zMid[0, iCell, k] = -bottomDepth[iCell] + \
                0.5 * layerThickness[0, iCell, k]
            for k in range(maxLevelCell[iCell] - 1, -1, -1):
                zMid[0, iCell, k] = zMid[0, iCell, k + 1] + 0.5 * \
                    (layerThickness[0, iCell, k + 1] + \
                    layerThickness[0, iCell, k])

        # linear equation of state
        # rho = rho0 - alpha*(T-Tref) + beta*(S-Sref)
        # set S=Sref
        # T = Tref - (rho - rhoRef)/alpha
        for k in range(0, nVertLevels):
            activeCells = k <= maxLevelCell
            salinity[0, activeCells, k] = config_eos_linear_Sref
            density[0, activeCells, k] = config_eos_linear_densityref + \
                (deltaRho/2)*(1.0 - np.tanh((2/interfaceThick)*np.arctanh(0.99)*(zMid[0, :, k] + \
                0.5*maxDepth - amplitude*np.cos(wavenumber*xCell[:]))))
            # T = Tref - (rho - rhoRef)/alpha
            temperature[0, activeCells, k] = config_eos_linear_Tref \
                - (density[0, activeCells, k] - config_eos_linear_densityref) / \
                config_eos_linear_alpha

        # initial velocity on edges
        ds['normalVelocity'] = (('Time', 'nEdges', 'nVertLevels',),
            np.zeros([1, nEdges, nVertLevels]))
        normalVelocity = ds['normalVelocity']
        for iEdge in range(0, nEdges):
            normalVelocity[0, iEdge, :] = u[0, iEdge, :] * math.cos(angleEdge[iEdge])

        # Coriolis parameter
        ds['fCell'] = (('nCells', 'nVertLevels',), np.zeros([nCells, nVertLevels]))
        ds['fEdge'] = (('nEdges', 'nVertLevels',), np.zeros([nEdges, nVertLevels]))
        ds['fVertex'] = (('nVertices', 'nVertLevels',),
            np.zeros([nVertices, nVertLevels]))

        # surface fields
        surfaceStress[:] = 0.0
        atmosphericPressure[:] = 0.0
        boundaryLayerDepth[:] = 0.0
        print('   time: %f' % ((time.time() - time1)))

        # comment('finalize and write file')
        time1 = time.time()
        ds['maxLevelCell'] = (['nCells'], maxLevelCell + 1)
        ds['restingThickness'] = (['nCells', 'nVertLevels'], restingThickness)
        for var in varsZ:
            ds[var] = (['nVertLevels'], globals()[var])
        for var in vars2D:
            ds[var] = (['nCells'], globals()[var])
        for var in vars3D:
            ds[var] = (['Time', 'nCells', 'nVertLevels'], globals()[var])
        # If you prefer not to have NaN as the fill value, you should consider
        # using mpas_tools.io.write_netcdf() instead
        ds.to_netcdf('initial_state.nc', format='NETCDF3_64BIT_OFFSET')
        print('   time: %f' % ((time.time() - time1)))
        print('Total time: %f' % ((time.time() - timeStart)))
