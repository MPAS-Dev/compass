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
import topo_builder
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
                        help='total depth'
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
    latEdge=ds['latEdge']
    latVertex=ds['latVertex']
    lonCell=ds['lonCell']

    # x values for convenience
    xMax = max(xCell)
    xMin = min(xCell)
    xMid = 0.5 * (xMin + xMax)

    # y values for convenience
    yMax = max(yCell)
    yMin = min(yCell)
    yMid = 0.5 * (yMin + yMax)
    rad2deg = 180.0/np.pi

    comment('create and initialize variables')
    time1 = time.time()

    varsZ = [ 'refLayerThickness', 'refBottomDepth', 'refZMid', 'vertCoordMovementWeights']
    for var in varsZ:
        globals()[var] = np.nan * np.ones(nVertLevels)

    vars2D = ['ssh', 'bottomDepth', 'bottomDepthObserved',
        'windStressZonal','windStressMeridional', 'atmosphericPressure', 'boundaryLayerDepth']
    for var in vars2D:
        globals()[var] = np.nan * np.ones(nCells)
    maxLevelCell = np.ones(nCells, dtype=np.int32)

    vars3D = [ 'layerThickness','temperature', 'salinity',
         'zMid', 'density']
    for var in vars3D:
        globals()[var] = np.nan * np.ones([1, nCells, nVertLevels])
    restingThickness = np.nan * np.ones([nCells, nVertLevels])

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

    # initialize topography
    ssh[:] = 0.0


    # NeverWorld2 domain
    NW2_lonW, NW2_lonE = 0, 60
    NW2_latS, NW2_latN = -70, 70
    
    D0 = 4000 # Nominal depth (m)
    cd = 200 # Depth of coastal shelf (m)
    drake = 2500 # Depth of Drake sill (m)
    cw = 5 # Width of coastal shelf (degrees)
    
    # Logical domain (grid points)
    # this is for structured grid: nj, ni = 140, 80
    # Simple "Atlantic" box with re-entrant Drake passage
    T = topo_builder.topo(lonCell*rad2deg, latCell*rad2deg, D0)
    T.add_NS_coast(NW2_lonW, -40, 90, cw, cd)
    T.add_NS_coast(NW2_lonE, -40, 90, cw, cd)
    T.add_NS_coast(NW2_lonW, -90, -60, cw, cd)
    T.add_NS_coast(NW2_lonE, -90, -60, cw, cd)
    T.add_EW_coast(-360, 360, NW2_latS, cw, cd)
    T.add_EW_coast(-360, 360, NW2_latN, cw, cd)
    bottomDepthObserved[:] = -T.z[:]
    print('Depth range: ', min(bottomDepthObserved), max(bottomDepthObserved))
    # Compute maxLevelCell and layerThickness for z-level (variation only on top)
    vertCoordMovementWeights[:] = 0.0
    vertCoordMovementWeights[0] = 1.0
    maxLevelCell[:] = 2
    bottomDepth[:] = refBottomDepth[2]
    print('bottomDepth range: ', min(bottomDepth), max(bottomDepth))
    for iCell in range(0, nCells):
        for k in range(nVertLevels-1, 0, -1):
            if bottomDepthObserved[iCell] > refBottomDepth[k - 1]:
                k = max(k, 2) #enforce minimun of 3 layers, i.e. 175m
                maxLevelCell[iCell] = k
                # Partial bottom cells
                bottomDepth[iCell] = max(bottomDepthObserved[iCell], refBottomDepth[2])
                # No partial bottom cells
                #bottomDepth[iCell] = refBottomDepth[k]
                layerThickness[0, iCell, k] = bottomDepth[iCell] - refBottomDepth[k - 1]
                break
        if bottomDepthObserved[iCell] <=  refBottomDepth[0]: 
           print('do you ever get in this loop?')
           k = 2 #enforce minimun of 3 layers, i.e. 175m
           maxLevelCell[iCell] = k
           bottomDepth[iCell] = refBottomDepth[k]
           layerThickness[0, iCell, k] = bottomDepth[iCell] - refBottomDepth[k - 1]
           #print(iCell, maxLevelCell[iCell],bottomDepth[iCell], layerThickness[0, iCell, 0:k+1] )
        # enforce minimum of 3 layers
        #maxLevelCell[iCell] = max(maxLevelCell[iCell],2)
        #bottomDepth[iCell] = max(bottomDepth[iCell],refBottomDepth[2])
        #if maxLevelCell[iCell]<2:
        #   print(iCell, maxLevelCell[iCell],bottomDepth[iCell])
        #print(iCell, maxLevelCell[iCell])
        #print(iCell, layerThickness[0, iCell, maxLevelCell[iCell] ], refLayerThickness[maxLevelCell[iCell]])
        layerThickness[0, iCell, 0:maxLevelCell[iCell] ] = refLayerThickness[0:maxLevelCell[iCell]]
        #print('after: ', iCell, layerThickness[0, iCell, 1], refLayerThickness[1])
        layerThickness[0, iCell, 0] += ssh[iCell]
        if bottomDepthObserved[iCell] <  refBottomDepth[2]:
           print('shallow: ', iCell,  maxLevelCell[iCell], layerThickness[0, iCell, 0: maxLevelCell[iCell]+1], refLayerThickness[0: maxLevelCell[iCell]+1])
           #print(iCell, layerThickness[0, iCell, 0] )
    print('bottomDepth range: ', min(bottomDepth), max(bottomDepth))
    #print('LayerThickness range: ', min(layerThickness), max(layerThickness))

    # Compute zMid (same, regardless of vertical coordinate)
    for iCell in range(0, nCells):
        k = maxLevelCell[iCell]
        zMid[0, iCell, k] = -bottomDepth[iCell] + \
            0.5 * layerThickness[0, iCell, k]
        for k in range(maxLevelCell[iCell] - 1, -1, -1):
            zMid[0, iCell, k] = zMid[0, iCell, k + 1] + 0.5 * \
                (layerThickness[0, iCell, k + 1] + layerThickness[0, iCell, k])
    restingThickness[:, :] = layerThickness[0, :, :]

    # Compute zMid (same, regardless of vertical coordinate)
    #for iCell in range(0, nCells):
    #    if abs(bottomDepth[iCell] - sum(layerThickness[0,iCell,0:maxLevelCell[iCell]+1]))>1.0:
    #        print(iCell,maxLevelCell[iCell],bottomDepth[iCell],sum(layerThickness[0,iCell,0:maxLevelCell[iCell]+1]))
    #        print(iCell,np.argmin(layerThickness[0,iCell,0:maxLevelCell[iCell]+1]), 'blab',min(layerThickness[0,iCell,0:maxLevelCell[iCell]+1]))

    # add tracers
    S0 = 35.0
    # equally spaced layers
    refDensity = np.zeros(nVertLevels)
    refDensity[:] = [1022.6, 1022.81, 1023.2, 1023.74, 1024.32, 1024.9, 1025.47, 1026.0, 1026.48, 1026.9, 1027.27, 1027.58, 1027.82, 1027.99, 1028.1]
    config_eos_linear_alpha = 0.2
    config_eos_linear_beta = 0.8
    config_eos_linear_Tref = 15.0
    config_eos_linear_Sref = 35.0
    config_eos_linear_densityref = 1026.0
    fCell = np.zeros([nCells])
    fEdge = np.zeros([nEdges])
    fCell = np.zeros([nCells])
    fCell = np.zeros([nCells])
    fEdge = np.zeros([nEdges])
    fVertex = np.zeros([nVertices])
    ds['fCell'] = (('nCells',), fCell)
    ds['fEdge'] = (('nEdges',), fEdge)
    ds['fVertex'] = (('nVertices',), fVertex)
	
## Alice to do: add realistic Coriolis as function of latitude.
    for iCell in range(0, nCells):
        fCell[iCell]=2.0*7.2921e-5*np.sin(latCell[iCell]) # numpy sin function takes angle in rad; 

    for iEdge in range(0, nEdges):
        fEdge[iEdge]=2.0*7.2921e-5*np.sin(latEdge[iEdge])

    for iVertex in range(0, nVertices):
        fVertex[iVertex]=2.0*7.2921e-5*np.sin(latVertex[iVertex])
    # For periodic domains, the max cell coordinate is also the domain width
    Lx = max(lonCell)
    Ly = max(latCell)
    # surface fields
    ytau = np.zeros(7)
    taud = np.zeros(7)
    ytau[:] = np.array([-70.,-45.,-15.,0.,15.,45.,70.])*np.pi/180.
    taud[:] = np.array([0,.2,-0.1,-.02,-.1,.1,0])
    for iCell in range(0, nCells):
        ks= min(max(0, bisect.bisect_right(ytau,latCell[iCell]) - 1), len(ytau)-2)
        #print('  ks : %f '%ks)
        #print('  calcwind : %f '%(taud[ks] + ( taud[ks+1] - taud[ks]) * scurve(latCell[iCell], ytau[ks], ytau[ks+1]-ytau[ks])))
        #ks = np.max(0, bisect.bisect_right(ytau,latCell[iCell]) - 1) #determine wind lat interval - only works for *sorted* ytau list
        #if latCell[iCell] > 0:
        #    windStressZonal[iCell] = 0.1
        #else:
        #    windStressZonal[iCell] = -0.1
        windStressZonal[iCell] = taud[ks] + ( taud[ks+1] - taud[ks]) * scurve(latCell[iCell], ytau[ks], ytau[ks+1]-ytau[ks])
    windStressMeridional[:] = 0.0
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
    ds['restingThickness'] = (['nCells', 'nVertLevels'], restingThickness)
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
