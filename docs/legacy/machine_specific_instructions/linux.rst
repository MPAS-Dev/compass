Personal Linux/OSX Machine
==========================

This approach worked for `xylar <http://github.com/xylar>`_ under Ubuntu 18.04
and `vanroekel <http://github.com/vanroekel>`_ under Max OS X 10.14.6.

Installation of MPAS dependencies including SCORPIO and the compass conda environment
-------------------------------------------------------------------------------------

First, run the following script in an empty directory that you can delete later:

.. code-block:: bash

    #!/bin/bash
    set -e

    export PNETCDF_VERSION=1.12.0
    export SCORPIO_VERSION=1.1.1

    # modify this to fit your system
    export CONDA_PATH=/home/xylar/miniconda3

    source ${CONDA_PATH}/etc/profile.d/conda.sh

    conda create -y -n mpas -c e3sm python=3.8 "compass=0.1.6=mpi_mpich*" \
        netcdf-fortran mpich fortran-compiler cxx-compiler c-compiler m4 git cmake

    conda activate mpas

    # modify this
    export PREFIX="${CONDA_PATH}/envs/mpas"

    export MPICC=mpicc
    export MPICXX=mpicxx
    export MPIF77=mpifort
    export MPIF90=mpifort
    export LDFLAGS="-L${PREFIX}/lib"

    rm -rf pnetcdf-${PNETCDF_VERSION}*

    wget https://parallel-netcdf.github.io/Release/pnetcdf-${PNETCDF_VERSION}.tar.gz

    tar xvf pnetcdf-${PNETCDF_VERSION}.tar.gz
    cd pnetcdf-${PNETCDF_VERSION}

    ./configure --prefix=${PREFIX}
    make
    make install

    cd ..

    rm -rf scorpio*

    git clone git@github.com:E3SM-Project/scorpio.git
    cd scorpio
    git checkout scorpio-v$SCORPIO_VERSION

    mkdir build
    cd build
    CC=mpicc FC=mpifort cmake -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DPIO_ENABLE_TIMING=OFF -DNetCDF_Fortran_PATH=$PREFIX \
        -DPnetCDF_Fortran_PATH=$PREFIX -DNetCDF_C_PATH=$PREFIX \
        -DPnetCDF_C_PATH=$PREFIX ..

    make
    make install

    cd ../..

Setup before compiling/running
------------------------------

Then, when you want to build or run MPAS-Ocean, source a file containing:


.. code-block:: bash

    conda activate mpas
    # Modify this path to point to your mpas conda environment
    export PREFIX="/home/xylar/miniconda3/envs/mpas"
    # this step might not be needed
    export MPAS_EXTERNAL_LIBS="-L${PREFIX}/lib -lnetcdff"
    export NETCDF=${PREFIX}
    export PNETCDF=${PREFIX}
    export PIO=${PREFIX}
    # change to one of the other cores as needed
    export CORE=ocean
    export AUTOCLEAN=true
