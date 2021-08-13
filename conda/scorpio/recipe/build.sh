#!/bin/bash

export NETCDF_C_PATH=$(dirname $(dirname $(which nc-config)))
export NETCDF_FORTRAN_PATH=$(dirname $(dirname $(which nf-config)))
export PNETCDF_PATH=$(dirname $(dirname $(which pnetcdf-config)))

mkdir build
cd build
FC=mpifort CC=mpicc CXX=mpicxx cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DPIO_USE_MALLOC=ON \
    -DPIO_ENABLE_TOOLS=OFF \
    -DPIO_ENABLE_TESTS=ON \
    -DPIO_ENABLE_TIMING=OFF \
    -DPIO_ENABLE_INTERNAL_TIMING=ON \
    -DNetCDF_C_PATH=$NETCDF_C_PATH \
    -DNetCDF_Fortran_PATH=$NETCDF_FORTRAN_PATH \
    -DPnetCDF_PATH=$PNETCDF_PATH ..

make

# make tests
# ctest

make install
