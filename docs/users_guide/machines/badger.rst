.. _machine_badger:

Badger
======

For details on LANL IC, see :ref:`machine_grizzly`.

login: ``ssh -t $my_username@wtrw.lanl.gov ssh ba-fe``


config options
--------------

Here are the default config options added when you choose ``-m badger`` when
setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /usr/projects/regionalclimate/COMMON_MPAS/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the slurm account
    account = e3sm

    # the number of multiprocessing or dask threads to use
    threads = 18

badger, intel
-------------

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/load_latest_compass.sh

    module purge
    module load cmake/3.16.2
    module load intel/19.0.4
    module load intel-mpi/2019.4
    module load friendly-testing
    module load hdf5-parallel/1.8.16
    module load pnetcdf/1.11.2
    module load netcdf-h5parallel/4.7.3
    module load mkl/2019.0.4

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/usr/projects/climate/SHARED_CLIMATE/compass/badger/compass-1.0.0/scorpio-1.1.6/intel/impi
    export ESMF=/usr/projects/climate/SHARED_CLIMATE/compass/badger/compass-1.0.0/esmf-8.1.0/intel/impi

    export AUTOCLEAN=true
    export USE_PIO2=true
    export HDF5_USE_FILE_LOCKING=FALSE

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi


badger, gnu
-----------

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/load_latest_compass.sh

    module purge
    module load cmake/3.16.2
    module load gcc/6.4.0
    module load mvapich2/2.3
    module load friendly-testing
    module load hdf5-parallel/1.8.16
    module load pnetcdf/1.11.2
    module load netcdf-h5parallel/4.7.3
    module load mkl/2019.0.4

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/usr/projects/climate/SHARED_CLIMATE/compass/badger/compass-1.0.0/scorpio-1.1.6/gnu/mvapich
    export ESMF=/usr/projects/climate/SHARED_CLIMATE/compass/badger/compass-1.0.0/esmf-8.1.0/gnu/mvapich

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
