import xarray
import numpy
import os
import shutil
import math
import time

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for the merry-go-round
    test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution, name='initial_state'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case
        """
        super().__init__(test_case=test_case, name=name)
        self.resolution = resolution

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'init.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        # TODO evaluate where speed up is needed for high resolution cases
        config = self.config
        logger = self.logger

        section = config['merry_go_round']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        temperature_background = section.getfloat('temperature_background')
        temperature_center = section.getfloat('temperature_center')
        salinity_background = section.getfloat('salinity_background')
        density_surface = section.getfloat('density_surface')
        density_vertical_gradient = \
            section.getfloat('density_vertical_gradient')
        tracer2_background = section.getfloat('tracer2_background')
        tracer3_background = section.getfloat('tracer3_background')

        section = config['vertical_grid']
        nz = section.getint('vertical_levels')
        vert_coord = section.get('coord_type')
        bottom_depth = section.getfloat('bottom_depth')

        # TODO consider changing this
        if not (vert_coord == 'z-level' or vert_coord == 'sigma'):
            print('Vertical coordinate {vert_coord} not supported')

        resolution = self.resolution
        res_params = {'5m':    {'nx': 100,
                                'ny': 4,
                                'nz': 50,
                                'dc': 5},
                      '2.5m':  {'nx': 200,
                                'ny': 4,
                                'nz': 100,
                                'dc': 2.5},
                      '1.25m': {'nx': 400,
                                'ny': 4,
                                'nz': 200,
                                'dc': 1.25}}
        res_params = res_params[resolution]
        dsMesh = make_planar_hex_mesh(nx=res_params['nx'],
                                      ny=res_params['ny'],
                                      dc=res_params['dc'],
                                      nonperiodic_x=True,
                                      nonperiodic_y=False)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = dsMesh.copy()
        angleEdge = ds.angleEdge.values
        xCell = ds.xCell
        yCell = ds.yCell
        xEdge = ds.xEdge
        xOffset = numpy.min(xEdge.values)
        xCellAdjusted = xCell - xOffset
        xEdgeAdjusted = xEdge - xOffset
        nCells = ds['nCells'].size
        nEdges = ds['nEdges'].size

        ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
        ds['ssh'] = xarray.zeros_like(xCell)

        # Note: machine-precision diffs in layerThickness are present from
        # legacy compass which did not use init_vertical_coord
        config.set('vertical_grid', 'vert_levels', str(res_params['nz']))
        init_vertical_coord(config, ds)

        nVertLevels = ds['nVertLevels'].size

        xMin = xCellAdjusted.min()
        xMax = xCellAdjusted.max()
        xMid = 0.5*(xMin + xMax)
        zMid = ds.zMid.values

        vertCoordMovementWeights = xarray.ones_like(ds.refZMid)
        if (vert_coord == 'z-level'):
            vertCoordMovementWeights[:] = 0.0
            vertCoordMovementWeights[0] = 1.0
        ds['vertCoordMovementWeights'] = vertCoordMovementWeights

        # Compute temperature
        xCellDepth, _ = xarray.broadcast(xCellAdjusted, ds.refBottomDepth)
        xCellAdjusted = xCellAdjusted.values
        xEdgeAdjusted = xEdgeAdjusted.values
        temperature = temperature_background * xarray.ones_like(xCellDepth)
        temperature = xarray.where(xCellDepth < xMid,
                                   temperature_center,
                                   temperature_background)
        temperature = temperature.expand_dims(dim='Time', axis=0)
        ds['temperature'] = temperature

        ds['salinity'] = salinity_background * xarray.ones_like(temperature)

        # Compute normalVelocity
        # identical with legacy at init
        xEdgeDepth, _ = xarray.broadcast(ds.xEdge, ds.refBottomDepth)
        normalVelocity = xarray.zeros_like(xEdgeDepth)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        for iEdge in range(0, nEdges):
            cell1 = ds.cellsOnEdge[iEdge, 0].values - 1
            cell2 = ds.cellsOnEdge[iEdge, 1].values - 1
            for k in range(0, nVertLevels):
                zMidEdge = 0.5*(zMid[0, cell1, k] + zMid[0, cell2, k])
                dPsi = - (2.0*zMidEdge + bottom_depth) / (0.5*bottom_depth)**2
                normalVelocity[iEdge, k] = (1.0 - (
                   (xEdgeAdjusted[iEdge] - 0.5 * xMax)**4) /
                   ((0.5*(xMax - xMin))**4)) * dPsi
                if ((cell1 < 0.5*nCells and cell1 > 0.25*nCells - 1) and
                    (cell2 < 0.5*nCells and cell2 > 0.25*nCells - 1)):
                    normalVelocity[iEdge, k] = (1.0 - (
                        ((xEdgeAdjusted[iEdge] - xMin) -
                         0.5*(xMax + xMin))**4) /
                        ((0.5*(xMax - xMin))**4)) * dPsi
                if (cell1 > 0.75*nCells-1 and cell2 > 0.75*nCells-1):
                    normalVelocity[iEdge, k] = (1.0 - (
                        ((xEdgeAdjusted[iEdge]-xMin) -
                         0.5*(xMax + xMin))**4) /
                        ((0.5*(xMax - xMin))**4)) * dPsi
            normalVelocity[iEdge, :] = \
                normalVelocity[iEdge, :] * math.cos(angleEdge[iEdge])
            if (xEdgeAdjusted[iEdge] <= xMin or xEdgeAdjusted[iEdge] >= xMax):
                normalVelocity[iEdge, :] = 0.0
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)
        ds['normalVelocity'] = normalVelocity

        tracer1 = xarray.zeros_like(temperature)
        # Note: order 1e-3 diffs are present compared with legacy
        for iCell in range(0, nCells):
            for k in range(0, nVertLevels):
                psi1 = 1.0 - (((xCellAdjusted[iCell] - 0.5*xMax)**4)/((0.5*xMax)**4))
                psi2 = 1.0 - (((zMid[0, iCell, k] + 0.5*bottom_depth)**2) /
                              ((0.5*bottom_depth)**2))
                psi = psi1*psi2
                tracer1[0, iCell, k] = 0.5*(1 + math.tanh(2*psi - 1))
        ds['tracer1'] = tracer1
        ds['tracer2'] = tracer2_background * xarray.ones_like(tracer1)
        ds['tracer3'] = tracer3_background * xarray.ones_like(tracer1)

        write_netcdf(ds, 'init.nc')
