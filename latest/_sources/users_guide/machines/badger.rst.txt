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

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_compass1.0.0_intel_impi.sh


To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi


badger, gnu
-----------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_compass1.0.0_gnu_mvapich.sh


To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran
