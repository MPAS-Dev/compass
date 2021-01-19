Anvil/Blues
===========

intel on anvil
--------------

First, you might want to build SCORPIO (see below), or use the one from
`xylar <http://github.com/xylar>`_ referenced here:

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

    module purge
    module load cmake/3.14.2-gvwazz3 intel/17.0.0-pwabdn2 \
        intel-mkl/2017.1.132-6qy7y5f netcdf/4.4.1-tckdgwl netcdf-cxx/4.2-3qkutvv \
        netcdf-fortran/4.4.4-urmb6ss mvapich2/2.2-verbs-qwuab3b \
        parallel-netcdf/1.11.0-6qz7skn
    module load git

    export NETCDF=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/netcdf-4.4.1-tckdgwl
    export NETCDFF=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/netcdf-fortran-4.4.4-urmb6ss
    export PNETCDF=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/parallel-netcdf-1.11.0-6qz7skn
    export PIO=/home/ac.xylar/libraries/scorpio-1.1.1-intel
    export AUTOCLEAN=true

    export I_MPI_CC=icc
    export I_MPI_CXX=icpc
    export I_MPI_F77=ifort
    export I_MPI_F90=ifort
    export USE_PIO2=true

    export MV2_ENABLE_AFFINITY=0
    export MV2_SHOW_CPU_BINDING=1

    export  HDF5_USE_FILE_LOCKING=FALSE

    make ifort CORE=ocean

SCORPIO on anvil
----------------

If you need to compile it yourself, you can do that as follows (contact
`xylar <http://github.com/xylar>`_ if you run into trouble):

.. code-block:: bash

    #!/bin/bash

    export SCORPIO_VERSION=1.1.1

    module purge
    module load cmake/3.14.2-gvwazz3 intel/17.0.0-pwabdn2 \
        intel-mkl/2017.1.132-6qy7y5f netcdf/4.4.1-tckdgwl netcdf-cxx/4.2-3qkutvv \
        netcdf-fortran/4.4.4-urmb6ss mvapich2/2.2-verbs-qwuab3b \
        parallel-netcdf/1.11.0-6qz7skn

    export NETCDF_C_PATH=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/netcdf-4.4.1-tckdgwl
    export NETCDF_FORTRAN_PATH=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/netcdf-fortran-4.4.4-urmb6ss
    export PNETCDF_PATH=/blues/gpfs/software/centos7/spack-latest/opt/spack/linux-centos7-x86_64/intel-17.0.0/parallel-netcdf-1.11.0-6qz7skn

    export SCORPIO_PATH=$HOME/libraries/scorpio-${SCORPIO_VERSION}-intel
    # export MPIROOT=$I_MPI_ROOT

    rm -rf scorpio*

    git clone git@github.com:E3SM-Project/scorpio.git
    cd scorpio
    git checkout scorpio-v$SCORPIO_VERSION

    mkdir build
    cd build
    FC=mpif90 CC=mpicc CXX=mpicxx cmake \
        -DCMAKE_INSTALL_PREFIX=$SCORPIO_PATH -DPIO_ENABLE_TIMING=OFF \
        -DNetCDF_C_PATH=$NETCDF_C_PATH -DNetCDF_Fortran_PATH=$NETCDF_FORTRAN_PATH \
        -DPnetCDF_PATH=$PNETCDF_PATH ..

    make
    make install
