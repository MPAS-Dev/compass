#!/usr/bin/env python
"""
This script writes a graph.info for base_mesh.nc from the build_mesh
routine.  This is needed for global cases that do not cull cells
"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import xarray
import os
from mpas_tools.conversion import convert
from mpas_tools.io import write_netcdf


write_netcdf(convert(xarray.open_dataset('base_mesh.nc'),
                     graphInfoFileName='graph.info'),
             'base_mesh_final.nc')

os.remove('base_mesh_final.nc')
