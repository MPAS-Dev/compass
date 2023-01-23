#!/bin/bash

set -xe

mkdir build
cd build

echo CMAKE_ARGS: ${CMAKE_ARGS}

FC=mpifort CC=mpicc CXX=mpicxx cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DPIO_USE_MALLOC:BOOL=ON \
    -DPIO_ENABLE_TOOLS:BOOL=OFF \
    -DPIO_ENABLE_TESTS:BOOL=ON \
    -DPIO_ENABLE_EXAMPLES:BOOL=OFF \
    -DPIO_ENABLE_TIMING:BOOL=OFF \
    -DWITH_HDF5:BOOL=OFF \
    -DWITH_NETCDF:BOOL=ON \
    -DWITH_PNETCDF:BOOL=ON \
    -DNetCDF_C_PATH=$PREFIX \
    -DNetCDF_Fortran_PATH=$PREFIX \
    -DPnetCDF_PATH=$PREFIX \
    ${SRC_DIR}

cmake --build .

ctest --output-on-failure

cmake --install .