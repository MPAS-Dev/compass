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

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /lcrc/soft/climate/compass/chrysalis/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

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



Intel on Chrysalis
------------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /lcrc/soft/climate/compass/chrysalis/load_latest_compass_intel_openmpi.sh

To build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] ifort


Gnu on Chrysalis
----------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /lcrc/soft/climate/compass/chrysalis/load_latest_compass_gnu_openmpi.sh

To build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gfortran

