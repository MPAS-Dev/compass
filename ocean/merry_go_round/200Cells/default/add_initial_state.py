#!/usr/bin/env python

'''
This script creates an initial condition file for MPAS-Ocean.

Merry-go-round Test Case: steady state test case for horizontal 
and vertical advection. 
see:
Calandrini, S., Pieper, K. and Gunzburger, M.D., 2020. Exponential time 
differencing for the tracer equations appearing in primitive equation 
ocean models. Computer Methods in Applied Mechanics and Engineering, 
365, p.113002.
'''

import os
import shutil
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
import argparse
import math
import time
verbose = True


def main():
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='base_mesh.nc',
                        help='Input file, containing base mesh'
                        )
    parser.add_argument('-o', '--output_file', dest='output_file',
                        default='initial_state.nc',
                        help='Output file, containing initial variables'
                        )
    parser.add_argument('-L', '--nVertLevels', dest='nVertLevels',
                        default=100,
                        help='Number of vertical levels'
                        )
    nVertLevels = parser.parse_args().nVertLevels
    ds = xr.open_dataset(parser.parse_args().input_file)

# configuration settings
    # reference vertical grid spacing
    maxDepth = 500
    vertical_coordinate = 'z'  # sigma or z

    # comment('obtain dimensions and mesh variables')
    nCells = ds['nCells'].size
    nEdges = ds['nEdges'].size
    nVertices = ds['nVertices'].size

    xCell = ds['xCell']
    xEdge = ds['xEdge']
    xVertex = ds['xVertex']
    yCell = ds['yCell']
    yEdge = ds['yEdge']
    yVertex = ds['yVertex']
    angleEdge = ds['angleEdge']
    cellsOnEdge = ds['cellsOnEdge']
    edgesOnCell = ds['edgesOnCell']

    # Adjust coordinates so first edge is at zero in x and y
    xOffset = xEdge.min()
    xCell -= xOffset
    xEdge -= xOffset
    xVertex -= xOffset
    yOffset = np.min(yEdge)
    yCell -= yOffset
    yEdge -= yOffset
    yVertex -= yOffset

    # x values for convenience
    xCellMax = max(xCell)
    xCellMin = min(xCell)
    xCellMid = 0.5 * (xCellMax + xCellMin)
    xEdgeMax = max(xEdge)
   
    # initialize velocity field 
    u = np.zeros([1, nEdges, nVertLevels])

    comment('create and initialize variables')
    time1 = time.time()

    varsZ = [ 'refLayerThickness', 'refBottomDepth', 'refZMid', 'vertCoordMovementWeights']
    for var in varsZ:
        globals()[var] = np.nan * np.ones(nVertLevels)

    vars2D = ['ssh', 'bottomDepth', 'surfaceStress',
	'atmosphericPressure', 'boundaryLayerDepth']
    for var in vars2D:
        globals()[var] = np.nan * np.ones(nCells)
    maxLevelCell = np.ones(nCells, dtype=np.int32)

    vars3D = [ 'layerThickness','temperature', 'salinity',
         'zMid', 'density', 'tracer1', 'tracer2', 'tracer3']
    for var in vars3D:
        globals()[var] = np.nan * np.ones([1, nCells, nVertLevels])
    restingThickness = np.nan * np.ones([nCells, nVertLevels])

    # Note that this line shouldn't be required, but if layerThickness is
    # initialized with nans, the simulation dies. It must multiply by a nan on
    # a land cell on an edge, and then multiply by zero.
    layerThickness[:] = -1e34

    # equally spaced layers
    refLayerThickness[:] = maxDepth / nVertLevels
    refBottomDepth[0] = refLayerThickness[0]
    refZMid[0] = -0.5 * refLayerThickness[0]
    for k in range(1, nVertLevels):
        refBottomDepth[k] = refBottomDepth[k - 1] + refLayerThickness[k]
        refZMid[k] = -refBottomDepth[k - 1] - 0.5 * refLayerThickness[k]

    # See surface height
    ssh[:] = 0.0

    # Compute maxLevelCell and layerThickness for z-level (variation only on top)
    if (vertical_coordinate=='z'):
        vertCoordMovementWeights[:] = 0.0
        vertCoordMovementWeights[0] = 1.0
        for iCell in range(0, nCells):
            maxLevelCell[iCell] = nVertLevels - 1
            bottomDepth[iCell] = refBottomDepth[nVertLevels - 1]
            layerThickness[0, iCell, :] = refLayerThickness[:]
            layerThickness[0, iCell, 0] += ssh[iCell]
        restingThickness[:, :] = layerThickness[0, :, :]

    # Compute maxLevelCell and layerThickness for sigma
    #elif (vertical_coordinate=='sigma'):
    #    vertCoordMovementWeights[:] = 1.0
    #    maxLevelCell[:] = nVertLevels - 1 # maxLevelCell is zero-based within python code
    #    bottomDepth[:] = bottomDepthObserved[:] 
    #    for iCell in range(0, nCells):
    #        restingThickness[iCell, :] = refLayerThickness[:]*bottomDepth[iCell]/maxDepth
    #        layerThickness[0, iCell, :] = refLayerThickness[:]*(ssh[iCell] + bottomDepth[iCell])/maxDepth

    # Compute zMid (same, regardless of vertical coordinate)
    for iCell in range(0, nCells):
        k = maxLevelCell[iCell]
        zMid[0, iCell, k] = -bottomDepth[iCell] + \
            0.5 * layerThickness[0, iCell, k]
        for k in range(maxLevelCell[iCell] - 1, -1, -1):
            zMid[0, iCell, k] = zMid[0, iCell, k + 1] + 0.5 * \
                (layerThickness[0, iCell, k + 1] + layerThickness[0, iCell, k])

    # initialize tracers
    rho0 = 1000.0  # kg/m^3
    rhoz = -2.0e-4  # kg/m^3/m in z
    S0 = 35.0
    r = 1 

    # linear equation of state
    # rho = rho0 - alpha*(T-Tref) + beta*(S-Sref)
    # set S=Sref
    # T = Tref - (rho - rhoRef)/alpha
    config_eos_linear_alpha = 0.2
    config_eos_linear_beta = 0.8
    config_eos_linear_Tref = 10.0
    config_eos_linear_Sref = 35.0
    config_eos_linear_densityref = 1000.0

    for k in range(0, nVertLevels):
        activeCells = k <= maxLevelCell
        salinity[0, activeCells, k] = S0
        density[0, activeCells, k] = rho0 + rhoz * zMid[0, activeCells, k]
        # T = Tref - (rho - rhoRef)/alpha
        #temperature[0, activeCells, k] = config_eos_linear_Tref \
        #    - (density[0, activeCells, k] - config_eos_linear_densityref) / \
        #      config_eos_linear_alpha
    for iCell in range(0, nCells):     
        if (xCell[iCell] < xCellMid): 
            temperature[0, iCell, :] = 5
        else: 
            temperature[0, iCell, :] = 30  
        for k in range(0, nVertLevels):
            psi1 = 1.0 - (((xCell[iCell] - 0.5*xCellMax)**4)/((0.5*xCellMax)**4))
            psi2 = 1.0 - (((zMid[0, iCell, k] + 0.5*maxDepth)**2)/((0.5*maxDepth)**2)) 
            psi = psi1*psi2
            tracer1[0,iCell,k] = 0.5*(1 + math.tanh( r*(2*psi - 1) ) )
    tracer2[0,:,:] = 10
    tracer3[0,:,:] = 20

    # initial velocity on edges
    ds['normalVelocity'] = (('Time', 'nEdges', 'nVertLevels',), np.zeros([1, nEdges, nVertLevels]))
    normalVelocity = ds['normalVelocity']
    for iEdge in range(0, nEdges):
        cell1 = cellsOnEdge[iEdge,0] - 1
        cell2 = cellsOnEdge[iEdge,1] - 1
        for k in range(0, nVertLevels):
            zMidEdge = 0.5*(zMid[0, cell1, k] + zMid[0, cell2, k])
            dPsi = - (2.0*zMidEdge + maxDepth) / (0.5*maxDepth)**2
            u[0, iEdge, k] = (1.0 - ((xEdge[iEdge] - 0.5*xCellMax)**4)/((0.5*(xCellMax-xCellMin))**4)) * dPsi
            if ((cell1<0.5*nCells and cell1>0.25*nCells-1) and (cell2<0.5*nCells and cell2>0.25*nCells-1)):
               u[0, iEdge, k] = (1.0 - (((xEdge[iEdge]-xCellMin) - 0.5*(xCellMax+xCellMin))**4)/((0.5*(xCellMax-xCellMin))**4)) * dPsi
            if (cell1>0.75*nCells-1 and cell2>0.75*nCells-1):
               u[0, iEdge, k] = (1.0 - (((xEdge[iEdge]-xCellMin) - 0.5*(xCellMax+xCellMin))**4)/((0.5*(xCellMax-xCellMin))**4)) * dPsi
        normalVelocity[0, iEdge, :] = u[0, iEdge, :] * math.cos(angleEdge[iEdge])
        if (xEdge[iEdge]<=xCellMin or xEdge[iEdge]>=xCellMax):
            normalVelocity[0, iEdge, :] = 0.0

    # Coriolis parameter
    ds['fCell'] = (('nCells', 'nVertLevels',), np.zeros([nCells, nVertLevels]))
    ds['fEdge'] = (('nEdges', 'nVertLevels',), np.zeros([nEdges, nVertLevels]))
    ds['fVertex'] = (('nVertices', 'nVertLevels',), np.zeros([nVertices, nVertLevels]))

    # surface fields
    surfaceStress[:] = 0.0
    atmosphericPressure[:] = 0.0
    boundaryLayerDepth[:] = 0.0
    print('   time: %f' % ((time.time() - time1)))

    comment('finalize and write file')
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
    # write_netcdf(ds,'initial_state.nc')
    print('   time: %f' % ((time.time() - time1)))
    print('Total time: %f' % ((time.time() - timeStart)))


def comment(string):
    if verbose:
        print('***   ' + string)


if __name__ == '__main__':
    # If called as a primary module, run main
    main()
