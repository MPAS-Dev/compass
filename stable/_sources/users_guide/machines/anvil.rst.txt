.. _machine_anvil:

Anvil
=====

Anvil is a set of nodes used by E3SM and its "ecosystem" projects on
``blues`` at `LCRC <https://www.lcrc.anl.gov/>`_.  To gain access to the
machine, you will need access to E3SM's confluence pages or the equivalent for
your ecosystem project.

config options
--------------

Here are the default config options added when you choose ``-m anvil`` when
setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /lcrc/soft/climate/e3sm-unified/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the number of multiprocessing or dask threads to use
    threads = 18

intel18 on anvil
----------------

First, you might want to build SCORPIO (see below), or use the one from
`xylar <http://github.com/xylar>`_ referenced here:

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

    module purge
    module load cmake/3.14.2-gvwazz3
    module load intel/18.0.4-62uvgmb
    module load intel-mkl/2018.4.274-jwaeshj
    module load netcdf/4.4.1-fijcsqi
    module load netcdf-cxx/4.2-cixenix
    module load netcdf-fortran/4.4.4-mmtrep3
    module load mvapich2/2.2-verbs-m57bia7
    module load parallel-netcdf/1.11.0-ny4vo3o

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/lcrc/soft/climate/compass/anvil/compass-1.0.0/scorpio-1.1.6/intel18/mvapich
    export ESMF=/lcrc/soft/climate/compass/anvil/compass-1.0.0/esmf-8.1.0/intel18/mvapich

    export I_MPI_CC=icc
    export I_MPI_CXX=icpc
    export I_MPI_F77=ifort
    export I_MPI_F90=ifort
    export MV2_ENABLE_AFFINITY=0
    export MV2_SHOW_CPU_BINDING=1

    export AUTOCLEAN=true
    export USE_PIO2=true
    export HDF5_USE_FILE_LOCKING=FALSE


To build the MPAS model with

.. code-block:: bash

    make CORE=landice ifort

or

.. code-block:: bash

    make CORE=ocean ifort

gnu on anvil
------------

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

    module purge
    module load cmake/3.14.2-gvwazz3
    module load gcc/8.2.0-xhxgy33
    module load intel-mkl/2018.4.274-2amycpi
    module load netcdf/4.4.1-ve2zfkw
    module load netcdf-cxx/4.2-2rkopdl
    module load netcdf-fortran/4.4.4-thtylny
    module load mvapich2/2.2-verbs-ppznoge
    module load parallel-netcdf/1.11.0-c22b2bn

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/lcrc/soft/climate/compass/anvil/compass-1.0.0/scorpio-1.1.6/gnu/mvapich
    export ESMF=/lcrc/soft/climate/compass/anvil/compass-1.0.0/esmf-8.1.0/gnu/mvapich

    export MV2_ENABLE_AFFINITY=0
    export MV2_SHOW_CPU_BINDING=1

    export AUTOCLEAN=true
    export USE_PIO2=true
    export HDF5_USE_FILE_LOCKING=FALSE

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran
