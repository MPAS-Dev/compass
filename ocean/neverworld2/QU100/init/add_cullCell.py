#!/usr/bin/env python
'''
This script adds the cullCell variable, which marks the cells to be culled.
'''
import os
import shutil
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
import argparse
import math
import time
verbose = True


def main():
    timeStart = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default='base_mesh.nc',
                        help='Input file, containing base mesh'
                        )
    ds = xr.open_dataset(parser.parse_args().input_file)

    #comment('obtain dimensions and mesh variables')
    nCells = ds['nCells'].size
    latCell = ds['latCell']
    lonCell = ds['lonCell']

# todo connect Drake passage - reentrant channel
    deg = np.pi/180.0
    cullCell = np.zeros(nCells, dtype=np.int32)
    cullCell[np.where( np.logical_or(
       latCell<-70*deg, np.logical_or( 
       latCell> 70*deg, np.logical_or(
       lonCell<  0*deg,
       lonCell> 60*deg
       ))))] = 1.0
    ds['cullCell'] = (['nCells'], cullCell)

    #ds.to_netcdf('initial_state.nc', format='NETCDF3_64BIT_OFFSET')
    write_netcdf(ds,parser.parse_args().input_file)
    print('Total time: %f' % ((time.time() - timeStart)))


def comment(string):
    if verbose:
        print('***   ' + string)


if __name__ == '__main__':
    # If called as a primary module, run main
    main()
