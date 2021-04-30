#!/usr/bin/env python

import xarray
import sys
import numpy


def compare_vars(var, ds1, ds2, mask):
    var1 = ds1[var].where(mask)
    var2 = ds2[var].where(mask)

    diff = numpy.abs(var1 - var2)
    if numpy.any(diff > 0.):
        print('  ERROR: {} not bit-for-bit:\n    {}'.format(
            var, diff.max().values))
        return False
    return True


def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    ds1 = xarray.open_dataset(filename1)
    ds2 = xarray.open_dataset(filename2)

    nCells = ds1.sizes['nCells']
    nVertLevels = ds1.sizes['nVertLevels']
    nEdges = ds1.sizes['nEdges']
    nVertices = ds1.sizes['nVertices']

    minLevelCell = ds2['minLevelCell'] - 1
    minLevelEdgeTop = ds2['minLevelEdgeTop'] - 1
    minLevelEdgeBot = ds2['minLevelEdgeBot'] - 1
    minLevelVertexTop = ds2['minLevelVertexTop'] - 1
    minLevelVertexBot = ds2['minLevelVertexBot'] - 1

    maxLevelCell = ds2['maxLevelCell'] - 1
    maxLevelEdgeTop = ds2['maxLevelEdgeTop'] - 1
    maxLevelEdgeBot = ds2['maxLevelEdgeBot'] - 1
    maxLevelVertexTop = ds2['maxLevelVertexTop'] - 1
    maxLevelVertexBot = ds2['maxLevelVertexBot'] - 1

    cellMask = xarray.DataArray(numpy.zeros((nCells, nVertLevels), dtype=bool),
                                dims=('nCells', 'nVertLevels'))

    edgeMaskInner = xarray.DataArray(
        numpy.zeros((nEdges, nVertLevels), dtype=bool),
        dims=('nEdges', 'nVertLevels'))

    edgeMaskOuter = xarray.DataArray(
        numpy.zeros((nEdges, nVertLevels), dtype=bool),
        dims=('nEdges', 'nVertLevels'))

    vertexMaskInner = xarray.DataArray(
        numpy.zeros((nVertices, nVertLevels), dtype=bool),
        dims=('nVertices', 'nVertLevels'))

    vertexMaskOuter = xarray.DataArray(
        numpy.zeros((nVertices, nVertLevels), dtype=bool),
        dims=('nVertices', 'nVertLevels'))

    for zIndex in range(nVertLevels):
        cellMask[:, zIndex] = numpy.logical_and(zIndex >= minLevelCell,
                                                zIndex <= maxLevelCell)

        edgeMaskInner[:, zIndex] = numpy.logical_and(
            zIndex >= minLevelEdgeBot, zIndex <= maxLevelEdgeTop)

        edgeMaskOuter[:, zIndex] = numpy.logical_and(
            zIndex >= minLevelEdgeTop, zIndex <= maxLevelEdgeBot)

        vertexMaskInner[:, zIndex] = numpy.logical_and(
            zIndex >= minLevelVertexBot, zIndex <= maxLevelVertexTop)

        vertexMaskOuter[:, zIndex] = numpy.logical_and(
            zIndex >= minLevelVertexTop, zIndex <= maxLevelVertexBot)

    success = True
    for iTime in range(ds2.sizes['Time']):
        print('iTime = {}'.format(iTime))
        print('  checking temperature')
        if not compare_vars('temperature', ds1.isel(Time=0),
                            ds2.isel(Time=iTime), cellMask):
            success = False

        print('  checking salinity')
        if not compare_vars('salinity', ds1.isel(Time=0),
                            ds2.isel(Time=iTime), cellMask):
            success = False

        print('  checking density')
        if not compare_vars('density', ds1.isel(Time=0),
                            ds2.isel(Time=iTime), cellMask):
            success = False

        print('  checking "inner" normal velocity')
        if not compare_vars('normalVelocity', ds1.isel(Time=0),
                            ds2.isel(Time=iTime), edgeMaskInner):
            success = False

        print('  checking "outer" normal velocity')
        if not compare_vars('normalVelocity', ds1.isel(Time=0),
                            ds2.isel(Time=iTime), edgeMaskOuter):
            success = False

    if success:
        print('PASS')
    else:
        print('FAIL')
        sys.exit(1)


if __name__ == '__main__':
    main()
