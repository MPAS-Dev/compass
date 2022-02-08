.. _machine_badger:

Badger
======

For details on LANL IC, see :ref:`machine_grizzly`.

login: ``ssh -t $my_username@wtrw.lanl.gov ssh ba-fe``


config options
--------------

Here are the default config options added when Badger is automatically
detected or you choose ``-m badger`` when setting up test cases or a test
suite:

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
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/compass/badger/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = impi

    # the system MPI library to use for gnu compiler
    mpi_gnu = mvapich

    # the base path to system libraries to be added as part of setting up compass
    system_libs = /usr/projects/climate/SHARED_CLIMATE/compass/badger/system

Additionally, some relevant config options come from the
`mache <https://github.com/E3SM-Project/mache/>`_ package:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # parallel system of execution: slurm, cobalt or single_node
    system = slurm

    # whether to use mpirun or srun to run a task
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # account for running diagnostics jobs
    account = e3sm

    # quality of service (default is the first)
    qos = regular, interactive

Intel on Badger
---------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_compass1.0.0_intel_impi.sh


To build the MPAS model with

.. code-block:: bash

    make intel-mpi

Gnu on Badger
-------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_compass1.0.0_gnu_mvapich.sh


To build the MPAS model with

.. code-block:: bash

    make gfortran
