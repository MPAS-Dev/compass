#!/usr/bin/env python

# This script was generated from setup_testcases.py as part of a config file

import sys
import os
import shutil
import glob
import subprocess


dev_null = open('/dev/null', 'w')

# Run command is:
# planar_hex --nx 1600 --ny 4 --dc 3e3 --nonperiodic_x -o planar_hex_mesh.nc
subprocess.check_call(['planar_hex', '--nx', '200', '--ny', '4', '--dc',
                       '2.5', '--nonperiodic_x', '-o', 'planar_hex_mesh.nc'])

# Run command is:
# MpasCellCuller.x planar_hex_mesh.nc culled_mesh.nc
subprocess.check_call(['MpasCellCuller.x', 'planar_hex_mesh.nc',
                       'culled_mesh.nc'])

# Run command is:
# MpasMeshConverter.x culled_mesh.nc base_mesh.nc
subprocess.check_call(['MpasMeshConverter.x', 'culled_mesh.nc', 'base_mesh.nc'])
