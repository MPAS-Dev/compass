#!/bin/bash

export NETCDF_C_PATH=$(dirname $(dirname $(which nc-config)))
export NETCDF_FORTRAN_PATH=$(dirname $(dirname $(which nf-config)))
export PNETCDF_PATH=$(dirname $(dirname $(which pnetcdf-config)))

mkdir build
cd build
# turning TESTS off temporarily because of a bug in 1.3.2
FC=mpifort CC=mpicc CXX=mpicxx cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DPIO_USE_MALLOC:BOOL=ON \
    -DPIO_ENABLE_TOOLS:BOOL=OFF \
    -DPIO_ENABLE_TESTS:BOOL=OFF \
    -DPIO_ENABLE_EXAMPLES:BOOL=OFF \
    -DPIO_ENABLE_TIMING:BOOL=OFF \
    -DPIO_ENABLE_INTERNAL_TIMING:BOOL=ON \
    -DNetCDF_C_PATH=$NETCDF_C_PATH \
    -DNetCDF_Fortran_PATH=$NETCDF_FORTRAN_PATH \
    -DPnetCDF_PATH=$PNETCDF_PATH ..

make

# make tests
# ctest

make install
