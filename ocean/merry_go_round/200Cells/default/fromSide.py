import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
matplotlib.use('Agg')

fig = plt.gcf()
fig.set_size_inches(8.0,10.0)

ncfile = Dataset('../forward/output/output.0001-01-01_00.00.00.nc', 'r')
var1 = ncfile.variables['velocityX']
var2 = ncfile.variables['vertVelocityTop']
var3 = ncfile.variables['tracer1']   
var11 = var1[:, 0:200, :]
var22 = var2[:, 0:200, :]
var33 = var3[:, 0:200, :]
print(var11[1,:,0])
for iCell in range(200):
    sumVert = 0.0
    for k in range(100):
        sumVert = sumVert + var11[1, iCell, k]
    print(sumVert)
print(var22[1,:,0])
for k in range(100):
    sumVert = 0.0
    for iCell in range(200):
        sumVert = sumVert + var22[1, iCell, k]
    print(sumVert)

plt.title('merry-go-round')
plt.subplot(2, 2, 1) 
ax = plt.imshow(var11[1, :, :].T)
plt.jet()
plt.xlabel('x, # of cells')
#plt.ylabel('depth, m')
plt.colorbar(ax, shrink=0.5)
plt.title('horizontal velocity')

plt.subplot(2, 2, 2)
ax = plt.imshow(var22[1, :, :].T)
plt.jet()
plt.xlabel('x, # of cells')
plt.colorbar(ax, shrink=0.5)
plt.title('vertical velocity')

plt.subplot(2, 2, 3)
ax = plt.imshow(var33[0, :, :].T)
plt.jet()
plt.xlabel('x, # of cells')
plt.colorbar(ax, shrink=0.5)
plt.title('tracer1 at t=0')

plt.subplot(2, 2, 4)
ax = plt.imshow(var33[1, :, :].T)
plt.jet()
plt.xlabel('x, # of cells')
plt.colorbar(ax, shrink=0.5)
plt.title('tracer1 at 6h')

ncfile.close()
plt.savefig('section.png')

