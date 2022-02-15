#!/usr/bin/env python
'''
This script plots results from MPAS-Ocean convergence test.
'''
import numpy as np
import math
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# mrp: read from file mesh after nx,ny attributes are added:
#nx = 10  # ncfileMesh.getncattr('nx')
#ny = 10  # ncfileMesh.getncattr('ny')
#iz = 5

dt = [12, 6, 3] # nRefinement
order2 = [6.4, 1.6, 0.4]
operators = ['tracer1']
nOperators = len(operators)

L2 = np.zeros([3])

for k in range(nOperators):
    ncfile12 = Dataset('../../../100Cells/default/forward/output' + '/output.0001-01-01_00.00.00.nc', 'r')
    ncfile6 = Dataset('../../../200Cells/default/forward/output' + '/output.0001-01-01_00.00.00.nc', 'r')
    ncfile3 = Dataset('../../../400Cells/default/forward/output' + '/output.0001-01-01_00.00.00.nc', 'r')
    #
    operator = operators[k]
    areas12 = ncfile12.variables['areaCell'][:]
    areas6 = ncfile6.variables['areaCell'][:]
    areas3 = ncfile3.variables['areaCell'][:]
    sol12 = ncfile12.variables[operator][1, :, 0]
    sol6 = ncfile6.variables[operator][1, :, 0]
    sol3 = ncfile3.variables[operator][1, :, 0]
    ref12 = ncfile12.variables[operator][0, :, 0]
    ref6 = ncfile6.variables[operator][0, :, 0]
    ref3 = ncfile3.variables[operator][0, :, 0]
    #
    dif12 = abs(sol12 - ref12)
    multDen12 = (ref12**2)*areas12
    multNum12 = (dif12**2)*areas12
    denL2_12 = np.sum(multDen12[:])/np.sum(areas12[:])
    numL2_12 = np.sum(multNum12[:])/np.sum(areas12[:])
    dif6 = abs(sol6 - ref6)
    multDen6 = (ref6**2)*areas6
    multNum6 = (dif6**2)*areas6
    denL2_6 = np.sum(multDen6[:])/np.sum(areas6[:])
    numL2_6 = np.sum(multNum6[:])/np.sum(areas6[:])
    dif3 = abs(sol3 - ref3)
    multDen3 = (ref3**2)*areas3
    multNum3 = (dif3**2)*areas3
    denL2_3 = np.sum(multDen3[:])/np.sum(areas3[:])
    numL2_3 = np.sum(multNum3[:])/np.sum(areas3[:])
    #
    L2[0] = np.sqrt(numL2_12)/np.sqrt(denL2_12)
    L2[1] = np.sqrt(numL2_6)/np.sqrt(denL2_6)
    L2[2] = np.sqrt(numL2_3)/np.sqrt(denL2_3)
    #
    order = math.log2(L2[0]/L2[1])
    print(order)
    order = math.log2(L2[1]/L2[2])
    print(order)
    #
    ncfile12.close()
    ncfile6.close()
    ncfile3.close()

for k in range(len(operators)):
    operator =operators[k]
    plt.loglog(dt, L2[:], '-x', label='rate')
plt.loglog(dt, order2, label='slope=-2')
plt.title('Convergence to the exact solution')
plt.ylabel('l_2 error norm')
plt.legend()
plt.grid()
plt.xticks(dt,dt)
plt.xlabel('time steps (in min)')

plt.savefig('order.png')

