#!/usr/bin/env python

import os
import xarray
import sys
import numpy


filename1 = sys.argv[1]
filename2 = sys.argv[2]
filename3 = filename2[:-3]+'_endtime.nc'

# Compare only the last time slice of filename2 with filename1. 
# Time dimension is length 3 in z_level test cases so min argument to -d is 2
os.system('ncks -d Time,2 '+filename2+' '+filename3)

ds1 = xarray.open_dataset(filename1)
ds2 = xarray.open_dataset(filename3)

var_list = list()
for var in ds1:
    if var in ds2:
        var_list.append(var)
    else:
        print('Warning: {} found in first but not second file'.format(var))

for var in ds2:
    if var not in ds1:
        print('Warning: {} found in second but not first file'.format(var))

all_pass = True
for var in var_list:
    var1 = ds1[var]
    var2 = ds2[var]

    if not numpy.issubdtype(var1.dtype, numpy.number) or \
            not numpy.issubdtype(var2.dtype, numpy.number):
        # not a numerical field
        continue

    if var1.sizes != var2.sizes:
        print('ERROR: {} not the same size: {} {}'.format(var, var1.sizes,
                                                          var2.sizes))
        all_pass = False
        continue

    diff = numpy.abs(var1 - var2)
    if numpy.any(diff > 0.):
        print('ERROR: {} not bit-for-bit:\n    {}'.format(var,
                                                          diff.max().values))
        all_pass = False

if all_pass:
    print('PASS')
else:
    print('FAIL')
    sys.exit(1)
