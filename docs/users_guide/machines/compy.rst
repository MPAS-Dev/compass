.. _machine_compy:

CompyMcNodeFace
===============

config options
--------------

Here are the default config options added when CompyMcNodeFace is automatically
detected or when you choose ``-m compy`` when setting up test cases or a test
suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /compyfs/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /compyfs/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /share/apps/E3SM/conda_envs/compass/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = impi

    # the system MPI library to use for gnu compiler
    mpi_pgi = mvapich2

    # the base path to system libraries to be added as part of setting up compass
    system_libs =  /share/apps/E3SM/conda_envs/compass/system

Additionally, some relevant config options come from the
`mache <https://github.com/E3SM-Project/mache/>`_ package:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # parallel system of execution: slurm, cobalt or single_node
    system = slurm

    # whether to use mpirun or srun to run a task
    parallel_executable = srun --mpi=pmi2

    # cores per node on the machine
    cores_per_node = 40

    # account for running diagnostics jobs
    account = e3sm

    # available partition(s) (default is the first)
    partitions = slurm

    # quality of service (default is the first)
    qos = regular


Intel on CompyMcNodeFace
------------------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source source /share/apps/E3SM/conda_envs/compass/load_compass1.0.0_intel_impi.sh

To build the MPAS model with

.. code-block:: bash

    make intel-mpi
