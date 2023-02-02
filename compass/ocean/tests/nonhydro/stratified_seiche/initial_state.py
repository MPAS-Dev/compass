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

        section = config['vertical_grid']
        maxDepth = section.getfloat('bottom_depth')
        nVertLevels = section.getint('vert_levels')
        section = config['stratified_seiche']
        config_eos_linear_alpha = section.getfloat('eos_linear_alpha')
        config_eos_linear_Tref = section.getfloat('eos_linear_Tref')
        config_eos_linear_Sref = section.getfloat('eos_linear_Sref')
        config_eos_linear_densityref = section.getfloat(
            'eos_linear_densityref')
        deltaRho = section.getfloat('deltaRho')
        interfaceThick = section.getfloat('interfaceThick')
        amplitude = section.getfloat('amplitude')
        wavenumber = section.getfloat('wavenumber')

        # comment('obtain dimensions and mesh variables')
        # vertical_coordinate = 'uniform'

        ds = dsMesh.copy()
        nCells = ds.nCells.size
        nEdges = ds.nEdges.size
        nVertices = ds.nVertices.size

        xCell = ds.xCell
        yCell = ds.yCell
        xEdge = ds.xEdge
        yEdge = ds.yEdge
        angleEdge = ds.angleEdge

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

        surfaceStress = np.nan * np.ones(nCells)
        atmosphericPressure = np.nan * np.ones(nCells)
        boundaryLayerDepth = np.nan * np.ones(nCells)

        ds['bottomDepth'] = maxDepth * xarray.ones_like(xCell)
        ds['ssh'] = xarray.zeros_like(xCell)

        init_vertical_coord(config, ds)

        # initial salinity, density, temperature
        ds['salinity'] = (config_eos_linear_Sref *
                          xarray.ones_like(ds.zMid)).where(ds.cellMask)
        ds['density'] = \
            (config_eos_linear_densityref +
             (deltaRho/2)*(1.0 - np.tanh(
                (2/interfaceThick)*np.arctanh(0.99) *
                (ds.zMid + 0.5*maxDepth -
                 amplitude*np.cos(wavenumber*ds.xCell)))))
        # T = Tref - (rho - rhoRef)/alpha
        ds['temperature'] = (config_eos_linear_Tref
                             - (ds.density - config_eos_linear_densityref) /
                             config_eos_linear_alpha)

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
        print(f'   time: {time.time() - time1}')

        # comment('finalize and write file')
        time1 = time.time()

        # If you prefer not to have NaN as the fill value, you should consider
        # using mpas_tools.io.write_netcdf() instead
        write_netcdf(ds, 'initial_state.nc')
        print(f'   time: {time.time() - time1}')
        print(f'Total time: {time.time() - timeStart}')
