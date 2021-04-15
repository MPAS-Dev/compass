.. _machine_chrysalis:

Chrysalis
=========

config options
--------------

Here are the default config options added when you choose ``-m chrysalis`` when
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
    cores_per_node = 64

    # the number of multiprocessing or dask threads to use
    threads = 18


intel on Chrysalis
------------------

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

    module purge
    module load subversion/1.14.0-e4smcy3
    module load perl/5.32.0-bsnc6lt
    module load intel/20.0.4-kodw73g
    module load intel-mkl/2020.4.304-g2qaxzf
    module load intel-mpi/2019.9.304-tkzvizk
    module load hdf5/1.8.16-se4xyo7
    module load netcdf-c/4.4.1-qvxyzq2
    module load netcdf-cxx/4.2-binixgj
    module load netcdf-fortran/4.4.4-rdxohvp
    module load parallel-netcdf/1.11.0-b74wv4m

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/lcrc/soft/climate/compass/chrysalis/compass-1.0.0/scorpio-1.1.6/intel/impi
    export ESMF=/lcrc/soft/climate/compass/chrysalis/compass-1.0.0/esmf-8.1.0/intel/impi

    export AUTOCLEAN=true
    export USE_PIO2=true
    export HDF5_USE_FILE_LOCKING=FALSE

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi


gnu on Chrysalis
----------------

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

    module purge
    module load subversion/1.14.0-e4smcy3
    module load perl/5.32.0-bsnc6lt
    module load gcc/9.2.0-ugetvbp
    module load intel-mkl/2020.4.304-n3b5fye
    module load openmpi/4.0.4-hpcx-hghvhj5
    module load hdf5/1.10.7-sbsigon
    module load netcdf-c/4.7.4-a4uk6zy
    module load netcdf-cxx/4.2-fz347dw
    module load netcdf-fortran/4.5.3-i5ah7u2
    module load parallel-netcdf/1.12.1-e7w4x32

    export NETCDF=$(dirname $(dirname $(which nc-config)))
    export NETCDFF=$(dirname $(dirname $(which nf-config)))
    export PNETCDF=$(dirname $(dirname $(which pnetcdf-config)))

    export PIO=/lcrc/soft/climate/compass/chrysalis/compass-1.0.0/scorpio-1.1.6/gnu/openmpi
    export ESMF=/lcrc/soft/climate/compass/chrysalis/compass-1.0.0/esmf-8.0.1/gnu/openmpi

    export AUTOCLEAN=true
    export USE_PIO2=true
    export HDF5_USE_FILE_LOCKING=FALSE

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran

