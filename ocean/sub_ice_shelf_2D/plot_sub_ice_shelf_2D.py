#!/usr/bin/env python

import xarray
import sys
import numpy
import matplotlib.pyplot as plt


filename = sys.argv[1]

ds = xarray.open_dataset(filename)

if 'Time' in ds.dims:
    ds = ds.isel(Time=0)

ds = ds.groupby('yCell').mean(dim='nCells')

nCells = ds.sizes['yCell']
nVertLevels = ds.sizes['nVertLevels']

zIndex = xarray.DataArray(data=numpy.arange(nVertLevels), dims='nVertLevels')

minLevelCell = ds.minLevelCell-1
maxLevelCell = ds.maxLevelCell-1

cellMask = numpy.logical_and(zIndex >= minLevelCell, zIndex <= maxLevelCell)

zIndex = xarray.DataArray(data=numpy.arange(nVertLevels+1), dims='nVertLevelsP1')

interfaceMask = numpy.logical_and(zIndex >= minLevelCell, zIndex <= maxLevelCell+1)

zTest = -ds.refBottomDepth.where(cellMask).values

zMid = ds.zMid.where(cellMask)

zInterface = numpy.zeros((nCells, nVertLevels+1))
zInterface[:, 0] = ds.ssh.values
for zIndex in range(nVertLevels):
    thickness = ds.layerThickness.isel(nVertLevels=zIndex)
    zInterface[:, zIndex+1] = zInterface[:, zIndex] - thickness.values

zInterface = xarray.DataArray(data=zInterface, dims=('yCell', 'nVertLevelsP1'))
zInterface = zInterface.where(interfaceMask)

plt.figure(figsize=[24, 12], dpi=100)
# plt.plot(ds.yCell.values, zTest, 'm')
plt.plot(ds.yCell.values, zMid.values, 'b')
plt.plot(ds.yCell.values, zInterface.values, 'k')
plt.plot(ds.yCell.values, ds.ssh.values, 'g')
plt.plot(ds.yCell.values, -ds.bottomDepth.values, 'g')
plt.savefig('vert_grid.png', dpi=200)
plt.show()
