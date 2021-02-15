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
zInterface = numpy.zeros((nCells, nVertLevels+1))
zInterface[:, 0] = ds.ssh.values
for zIndex in range(nVertLevels):
    zInterface[:, zIndex+1] = zInterface[:, zIndex] - \
                              ds.layerThickness.isel(nVertLevels=zIndex).values

plt.figure()
plt.plot(ds.yCell, zInterface, 'k')
plt.plot(ds.yCell, ds.zMid, 'b')
plt.show()
