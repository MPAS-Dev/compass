.. _machine_chrysalis:

Chrysalis
=========

config options
--------------

Here are the default config options added when Chrysalis is automatically
detected or when you choose ``-m chrysalis`` when setting up test cases or a
test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # A shared root directory where MPAS standalone data can be found
    database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata

    # the path where shared compass environments are deployed
    compass_envs = /lcrc/soft/climate/compass/chrysalis/base


    # Options related to deploying compass environments on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the compiler to use to build software (e.g. ESMF and MOAB) with spack
    software_compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = openmpi

    # the system MPI library to use for gnu compiler
    mpi_gnu = openmpi

    # the base path for spack environments used by compass
    spack = /lcrc/soft/climate/compass/chrysalis/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    use_e3sm_hdf5_netcdf = True

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
    cores_per_node = 128

    # available partition(s) (default is the first)
    partitions = debug, compute, high



Loading and running compass on Chrysalis
----------------------------------------

Follow the Developer's Guide at :ref:`dev_machine_chrysalis` to deploy
``compass`` and build MPAS components.  There are currently no shared
``compass`` environments for users on Chrysalis.
