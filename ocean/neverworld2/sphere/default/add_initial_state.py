#!/usr/bin/env python
'''
This script creates an initial condition file for MPAS-Ocean.
'''
import os
import shutil
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
import argparse
import math
import time
import bisect
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
                        default=15,
                        help='Number of vertical levels'
                        )
    parser.add_argument('-H', '--maxDepth', dest='maxDepth',
                        default=4000,
                        help='Number of vertical levels'
                        )
    nVertLevels = int(parser.parse_args().nVertLevels)
    maxDepth = float(parser.parse_args().maxDepth)
    ds = xr.open_dataset(parser.parse_args().input_file)

    #comment('obtain dimensions and mesh variables')
    nCells = ds['nCells'].size
    nEdges = ds['nEdges'].size
    nVertices = ds['nVertices'].size

    xCell = ds['xCell']
    xEdge = ds['xEdge']
    xVertex = ds['xVertex']
    yCell = ds['yCell']
    yEdge = ds['yEdge']
    yVertex = ds['yVertex']
    latCell=ds['latCell']
    lonCell=ds['lonCell']

    # x values for convenience
    xMax = max(xCell)
    xMin = min(xCell)
    xMid = 0.5 * (xMin + xMax)

    # y values for convenience
    yMax = max(yCell)
    yMin = min(yCell)
    yMid = 0.5 * (yMin + yMax)

    comment('create and initialize variables')
    time1 = time.time()

    varsZ = [ 'refLayerThickness', 'refBottomDepth', 'refZMid', 'vertCoordMovementWeights']
    for var in varsZ:
        globals()[var] = np.nan * np.ones(nVertLevels)

    vars2D = ['ssh', 'bottomDepth', 'bottomDepthObserved',
        'surfaceStress','windStressZonal','windStressMeridional', 'atmosphericPressure', 'boundaryLayerDepth']
    for var in vars2D:
        globals()[var] = np.nan * np.ones(nCells)
    maxLevelCell = np.ones(nCells, dtype=np.int32)

    vars3D = [ 'layerThickness','temperature', 'salinity',
         'restingThickness', 'zMid', 'density']
    for var in vars3D:
        globals()[var] = np.nan * np.ones([1, nCells, nVertLevels])

    # Note that this line shouldn't be required, but if layerThickness is
    # initialized with nans, the simulation dies. It must multiply by a nan on
    # a land cell on an edge, and then multiply by zero.
    layerThickness[:] = -1e34

    # equally spaced layers
    refLayerThickness[:] = [25.0, 50.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 300.0, 350.0, 400.0, 500.0, 550.0, 600.0] #maxDepth / nVertLevels
    refBottomDepth[0] = refLayerThickness[0]
    refZMid[0] = -0.5 * refLayerThickness[0]
    for k in range(1, nVertLevels):
        refBottomDepth[k] = refBottomDepth[k - 1] + refLayerThickness[k]
        refZMid[k] = -refBottomDepth[k - 1] - 0.5 * refLayerThickness[k]

    # Gaussian function in depth for deep sea ridge
    bottomDepthObserved[:] = maxDepth
    ssh[:] = 0.0

    # Compute maxLevelCell and layerThickness for z-level (variation only on top)
    vertCoordMovementWeights[:] = 0.0
    vertCoordMovementWeights[0] = 1.0
    for iCell in range(0, nCells):
        for k in range(nVertLevels - 1, 0, -1):
            if bottomDepthObserved[iCell] > refBottomDepth[k - 1]:

                maxLevelCell[iCell] = k
                # Partial bottom cells
                bottomDepth[iCell] = bottomDepthObserved[iCell]
                # No partial bottom cells
                #bottomDepth[iCell] = refBottomDepth[k]

                layerThickness[0, iCell, k] = bottomDepth[iCell] - refBottomDepth[k - 1]
                break
        layerThickness[0, iCell, 0:maxLevelCell[iCell] ] = refLayerThickness[0:maxLevelCell[iCell]]
        layerThickness[0, iCell, 0] += ssh[iCell]

    # Compute zMid (same, regardless of vertical coordinate)
    for iCell in range(0, nCells):
        k = maxLevelCell[iCell]
        zMid[0, iCell, k] = -bottomDepth[iCell] + \
            0.5 * layerThickness[0, iCell, k]
        for k in range(maxLevelCell[iCell] - 1, -1, -1):
            zMid[0, iCell, k] = zMid[0, iCell, k + 1] + 0.5 * \
                (layerThickness[0, iCell, k + 1] + layerThickness[0, iCell, k])
    restingThickness[:, :] = layerThickness[0, :, :]
    restingThickness[:, 0] = refLayerThickness[0]

    # add tracers
    T0 = 10.0 # Change to add T vertical profile (here or below)
    S0 = 35.0
    for k in range(0, nVertLevels):
        activeCells = k <= maxLevelCell
        salinity[0, activeCells, k] = S0
        temperature[0, activeCells, k] = T0

    # initial velocity on edges
    ds['normalVelocity'] = (('Time', 'nEdges', 'nVertLevels',), np.zeros([1, nEdges, nVertLevels]))

    # Coriolis parameter
# Nairita, add f0+beta here
    fCell = np.zeros([nCells])
    fEdge = np.zeros([nEdges])
    fVertex = np.zeros([nVertices])
    ds['fCell'] = (('nCells',), fCell)
    ds['fEdge'] = (('nEdges',), fEdge)
    ds['fVertex'] = (('nVertices',), fVertex)
	
    for iCell in range(0, nCells):
        fCell[iCell]=2.0*7.29e-5

    for iEdge in range(0, nEdges):
        fEdge[iEdge]=2.0*7.29e-5

    for iVertex in range(0, nVertices):
        fVertex[iVertex]=2.0*7.29e-5
    # surface fields
    # Reference values of surface zonal wind stress
    ytau,taud = [-70,-45,-15,0,15,45,70], [0,.2,-0.1,-.02,-.1,.1,0]
# Nairita, add surface stress here
    ds['windStressZonal'] = (('nCells',), np.zeros([nCells,]))
    ds['windStressMeridional'] = (('nCells',), np.zeros([nCells,]))
    #lonCell = ds.variables['lonCell']
    #latCell = ds.variables['latCell']
    # For periodic domains, the max cell coordinate is also the domain width
    Lx = max(lonCell)
    Ly = max(latCell)

    for iCell in range(0, nCells):
        x = xCell[iCell]
        y = yCell[iCell]
        ks = np.min(0, bisect.bisect_right(ytau,y) - 1) #determine wind lat interval - only works for *sorted* ytau list
	#layerThickness[0, iCell,:] = np.exp(-(x-Lx/2.0)**2.0-(y-Ly/2.0)**2.0)
        windStressZonal[iCell] = taud[ks] + ( taud[ks+1] - taud[ks]) * scurve(y, ytau[ks], ytau[ks+1]-ytau[ks])

    #surfaceStress[:] = 0.0
    atmosphericPressure[:] = 0.0
    boundaryLayerDepth[:] = 0.0
    print('   time: %f' % ((time.time() - time1)))

    comment('finalize and write file')
    time1 = time.time()
    ds['maxLevelCell'] = (['nCells'], maxLevelCell + 1)
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



def scurve(x, x0, dx):
    """Returns 0 for x<x0 or x>x+dx, and a cubic in between."""
    s = np.minimum(1, np.maximum(0, (x-x0)/dx))
    return (3 - 2*s)*( s*s )


def comment(string):
    if verbose:
        print('***   ' + string)


if __name__ == '__main__':
    # If called as a primary module, run main
    main()
