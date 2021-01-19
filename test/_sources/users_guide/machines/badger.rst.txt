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

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/mesh_database
    initial_condition_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/initial_condition_database
    bathymetry_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/bathymetry_database

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

    # the number of multiprocessing or dask threads to use
    threads = 18


badger, gnu
-----------

.. code-block:: bash

    module use /usr/projects/climate/SHARED_CLIMATE/modulefiles/spack-lmod/linux-rhel7-x86_64

    # IC mods
    module load gcc/6.4.0
    module load openmpi/2.1.2
    module load cmake/3.12.1
    module load mkl

    # spack mods
    module load openmpi/2.1.2-bheb4xe/gcc/6.4.0/netcdf/4.4.1.1-zei2j6r
    module load openmpi/2.1.2-bheb4xe/gcc/6.4.0/netcdf-fortran/4.4.4-v6vwmxs
    module load openmpi/2.1.2-bheb4xe/gcc/6.4.0/parallel-netcdf/1.8.0-2qwcdbn
    module load openmpi/2.1.2-bheb4xe/gcc/6.4.0/pio/1.10.0-ljj73au

    export NETCDF=/usr/projects/climate/SHARED_CLIMATE/software/badger/spack-install/linux-rhel7-x86_64/gcc-6.4.0/netcdf-fortran-4.4.4-v6vwmxsv33t7pmulojlijwdbikrvmwkc
    export PNETCDF=/usr/projects/climate/SHARED_CLIMATE/software/badger/spack-install/linux-rhel7-x86_64/gcc-6.4.0/parallel-netcdf-1.8.0-2qwcdbnjcq5pnkoqpx2s7um3s7ffo3xd
    export PIO=/usr/projects/climate/SHARED_CLIMATE/software/badger/spack-install/linux-rhel7-x86_64/gcc-6.4.0/pio-1.10.0-ljj73au6ctgkwmh3gbd4mleljsumijys/

    make gfortran CORE=ocean
