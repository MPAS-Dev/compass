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

    # A shared root directory where MPAS standalone data can be found
    database_root = /compyfs/mpas_standalonedata

    # the path where shared compass environments are deployed
    compass_envs = /share/apps/E3SM/conda_envs/compass/base


    # Options related to deploying compass environments on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the compiler to use to build software (e.g. ESMF and MOAB) with spack
    software_compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = impi

    # the system MPI library to use for gnu compiler
    mpi_gnu = openmpi

    # the base path for spack environments used by compass
    spack = /share/apps/E3SM/conda_envs/compass/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    #
    # We don't use them on Compy because hdf5 and netcdf were build without MPI
    use_e3sm_hdf5_netcdf = False

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


Loading and running compass on CompyMcNodeFace
----------------------------------------------

Follow the Developer's Guide at :ref:`dev_machine_compy` to deploy
``compass`` and build MPAS components.  There are currently no shared
``compass`` environments for users on CompyMcNodeFace.
