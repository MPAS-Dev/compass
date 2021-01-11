Personal OSX Machine
====================

This personal approach worked for macOS Catalina 10.15.

Required: `Homebrew <https://brew.sh>`_

Installation of MPAS dependencies except PIO
--------------------------------------------

.. code-block:: bash

    git clone https://github.com/pwolfram/homebrew-mpas.git
    cd homebrew-mpas
    vi install.sh
    #    Comment out line 19 (#brew install pwolfram/mpas/pio --build-from-source)
    chmod 700 install.sh
    ./install.sh
    cd ..

PIO-1.9.23 installation with some modifications
-----------------------------------------------

.. code-block:: bash

    wget https://github.com/NCAR/ParallelIO/archive/pio1_9_23.tar.gz
    tar xzvf pio1_9_23.tar.gz
    cd ParallelIO-pio1_9_23
    cd pio
    git clone https://github.com/PARALLELIO/genf90.git bin
    git clone https://github.com/CESM-Development/CMake_Fortran_utils.git cmake
    vi pio_types.F90
    #      Go to line 309
    #             Change 'nf_max_var_dims' to '6'  (i.e., PIO_MAX_VAR_DIMS = 6)
    #      Go to line 328
    #             Change 'nf_max_var_dims' to '6'
    cd ..
    mkdir build
    cd build
    # Set shell environmental variables (for BASH)
    export FC=mpif90
    export CC=mpicc
    cmake ../
    make
    # PIO libs and includes will be installed in ParallelIO-pio1_9_23/build/pio
    cd ../../

MPAS-O installation
-------------------

.. code-block:: bash

    git clone https://github.com/MPAS-Dev/MPAS-Model.git
    cd MPAS-Model

    # Set shell environmental variables (for BASH)
    export PIO="PATH_TO_PIO_INSTALL"
    # example:  export PIO="/Users/3hk/test/ParallelIO-pio1_9_23/build/pio"
    export NETCDF="PATH_TO_NETCDF_INSTALL"
    # example:  export NETCDF="/usr/local/Cellar/netcdf/4.6.3_1"
    export NETCDFF="PATH_TO_NETCDFF_INSTALL"
    # example:  export NETCDFF="/usr/local/Cellar/netcdf/4.6.3_1"
    export PNETCDF="PATH_TO_PNETCDF_INSTALL"
    # example:  export PNETCDF="/usr/local/Cellar/parallel-netcdf/1.7.0_2"
    make gfortran CORE=ocean
    # or
    make gfortran-clang CORE=ocean