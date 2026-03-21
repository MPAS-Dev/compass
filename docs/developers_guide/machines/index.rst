.. _dev_machines:

Machines
========

One of the major advantages of ``compass`` over :ref:`legacy_compass` is that it
attempts to be aware of the capabilities of the machine it is running on.  This
is a particular advantage for so-called "supported" machines with a config file
defined for them in the ``compass`` package.  But even for "unknown" machines,
it is not difficult to set a few config options in your user config file to
describe your machine.  Then, ``compass`` can use this data to make sure test
cases are configured in a way that is appropriate for your machine.

.. _dev_supported_machines:

Supported Machines
------------------

If you follow the procedure in :ref:`dev_conda_env`, you will have one or more
generated load scripts in the root of your compass branch.  Source the one
that matches the machine, compiler and MPI library you want to use, e.g.:

.. code-block:: bash

    source load_compass_anvil_intel_impi.sh

After loading this environment, you can set up test cases or test suites, and
a link ``load_compass_env.sh`` will be included in each suite or test case
work directory.  This is a link to the activation script that you sourced when
you were setting things up.  You can can source this file on a compute node
(e.g. in a job script) to get the right compass deployment environment,
compilers, MPI libraries and environment variables for running ``compass``
tests and the MPAS model.

.. note::

  Albany (and therefore most of the functionality in MALI) is currently only
  supported for those configurations with ``gnu`` compilers.


+--------------+------------+-----------+-------------------+
| Machine      | Compiler   | MPI lib.  |  MPAS make target |
+==============+============+===========+===================+
| anvil        | intel      | impi      | intel-mpi         |
|              |            +-----------+-------------------+
|              |            | openmpi   | ifort             |
|              +------------+-----------+-------------------+
|              | gnu        | openmpi   | gfortran          |
|              |            +-----------+-------------------+
|              |            | mvapich   | gfortran          |
+--------------+------------+-----------+-------------------+
| chicoma-cpu  | gnu        | mpich     | gnu-cray          |
+--------------+------------+-----------+-------------------+
| chrysalis    | intel      | openmpi   | ifort             |
|              +------------+-----------+-------------------+
|              | gnu        | openmpi   | gfortran          |
+--------------+------------+-----------+-------------------+
| compy        | intel      | impi      | intel-mpi         |
+--------------+------------+-----------+-------------------+
| pm-cpu       | gnu        | mpich     | gnu-cray          |
+--------------+------------+-----------+-------------------+

Below are specifics for each supported machine

.. toctree::
   :titlesonly:

   anvil
   chicoma
   chrysalis
   compy
   perlmutter


.. _dev_other_machines:

Other Machines
--------------

If you are working on an unknown machine, the procedure is pretty similar to
what was described in :ref:`dev_conda_env`.  In general, use ``./deploy.py``
to create a local pixi environment and load scripts.  For example, on Linux
run:

.. code-block:: bash

    ./deploy.py --no-spack --compiler gnu --mpi mpich

and on OSX run:

.. code-block:: bash

    ./deploy.py --no-spack --compiler clang --mpi mpich

You may use ``openmpi`` instead of ``mpich`` but we have had better experiences
with the latter.

The result should be one or more generated load scripts.  Source the one that
matches the environment you want to use to get the appropriate deployment
environment and environment variables.

Under Linux, you can build the MPAS model with

.. code-block:: bash

    make gfortran

Under OSX, you can build the MPAS model with

.. code-block:: bash

    make gfortran-clang

.. _dev_add_supported_machine:

Adding a New Supported Machine
------------------------------

If you want to add a new supported machine, you need to add a config file in
``compass`` and corresponding machine support in ``mache``.

Adding a Machine Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step in adding a new supported machine is to add a config file in
``compass/machines``.  The config file needs to describe the parallel
environment and some paths where shared Spack environments will be installed
and shared data will be downloaded.  The easiest place to start is one of the
examples provided (machines ``morpheus`` and ``eligos`` for now, but more
will be added soon.)

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # parallel system of execution: slurm, cobalt or single_node
    system = single_node

    # whether to use mpirun or srun to run a task
    parallel_executable = mpirun

    # cores per node on the machine
    cores_per_node = 8


    # Config options related to spack environments
    [spack]

    # whether to load modules from the spack yaml file before loading the spack
    # environment
    modules_before = False

    # whether to load modules from the spack yaml file after loading the spack
    # environment
    modules_after = False


    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # A shared root directory where MPAS standalone data can be found
    database_root = /home/xylar/data/mpas/mpas_standalonedata

    # the path where deployed compass environments are located
    compass_envs = /home/xylar/data/mpas/compass_envs


    # Options related to deploying compass environments on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = gnu

    # the system MPI library to use for gnu compiler
    mpi_gnu = openmpi

    # the base path for spack environments used by compass
    spack = /home/xylar/data/mpas/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    use_e3sm_hdf5_netcdf = False


    # Options related to machine discovery
    [discovery]

    # a substring used to identify this machine from its hostname
    hostname_contains = morpheus


The ``[parallel]`` section should describe the type of parallel queuing
system (currently only ``slurm`` or ``single_node`` are supported), the number
of cores per node and the command for running an MPI executable (typically
``srun`` for Slurm and ``mpirun`` for a "single node" machine like a laptop or
workstation.

The ``[spack]`` section has some config options to do with loading system
modules before or after loading a Spack environment.  On a "single node"
machine, you typically don't have modules so both ``modules_before`` and
``modules_after`` can be set to ``False``.  On a high-performance computing
(HPC) machine, you may find it is safest to load modules after the Spack
environment to ensure that certain paths and environment variables are set the
way the modules have them, rather than the way that Spack would have them.
The recommended starting point would be ``modules_before = False`` and
``modules_after = True``, but could be adjusted as needed if the right shared
libraries aren't being found when you try to build an MPAS component.

In the ``[paths]`` section, you will first give a path where you would like
to store shared data files used in compass test cases in ``database_root``.
Compass will create this directory if it doesn't exist.  Then, you can specify
``compass_envs`` as a path where shared deployment environments will be
installed for compass releases.  If developers always create their own local
environments, this path will never be used.

In ``[deploy]``, you will specify config options used in setting up deployment
and Spack environments for developers.  The ``compiler`` is the default
compiler to use for your system.  You must supply a corresponding
``mpi_<compiler>`` for each supported compiler (not just the default compiler)
that specifies the default MPI library for that compiler.  If you only support
one compiler and MPI library, that's pretty simple: ``compiler`` is the name
of the compiler (e.g. ``intel`` or ``gnu``) and ``mpi_<compiler>`` is the
MPI library (e.g. ``compiler_gnu = mpich`` or ``compiler_intel = openmpi``).
The ``spack`` option specifies a path where Spack environment will be created.
The option ``use_e3sm_hdf5_netcdf = False`` indicates that you will not use
the E3SM default modules for HDF5 and NetCDF libraries (which are not available
for machines installed in the way described here).

Finally, ``[discovery]`` allows you to add a ``hostname_contains`` that is used
to automatically identify your machine based on its hostname.  If your machine
has multiple login nodes with different hostnames, hopefully, a string common
to all login nodes can be used here.  If your machine has a unique hostname,
simply give that.  This option saves developers from having to specify
``--machine <machine>`` each time they deploy compass environments or set up
test cases.


Describing a Spack Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compass no longer carries machine-specific Spack environment templates inside
this repository.  Those templates now live in ``mache`` and are consumed by
``mache.deploy`` during ``./deploy.py`` runs.

When adding support for a new machine, the usual split is:

* add the Compass machine config in ``compass/machines``

* add or update the corresponding machine-specific Spack template in
  ``mache``

* keep the Compass package list in ``deploy/spack.yaml.j2`` in sync with the
  capabilities of that ``mache`` template

For examples from supported machines, compilers and MPI libraries, see the
`mache spack directory <https://github.com/E3SM-Project/mache/tree/main/mache/spack>`_.

Building the Spack Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to try setting up Compass and asking it to build the Spack
environment with a command something like:

.. code-block:: bash

    ./deploy.py --deploy-spack --compiler gnu --mpi openmpi --recreate

The ``--deploy-spack`` flag tells Compass to create or update supported Spack
environments.  You can specify a directory for testing Spack with the
``--spack-path`` flag.  If you need a custom temporary directory for Spack
builds, set ``spack.tmpdir`` in ``deploy/config.yaml.j2``.


Creating the Spack environment may take anywhere from minutes to hours,
depending on your system.

If dependencies don't build as expected, you may get an error message
suggesting that your ``operating_system`` or ``target`` aren't right. Here's
an example:

.. code-block::

    ==> Error: concretization failed for the following reasons:

       1. zlib compiler '%gcc@9.4.0' incompatible with 'os=ubuntu20.04'
       2. readline compiler '%gcc@9.4.0' incompatible with 'os=ubuntu20.04'
       3. pkgconf compiler '%gcc@9.4.0' incompatible with 'os=ubuntu20.04'

In this example, I had specified ``operating_system: ubuntu22.04`` in the YAML
file but in fact my operating system is ``ubuntu20.04`` as shown in the error
message.

You can run:

.. code-block:: bash

    source $SPACKDIR/share/spack/setup-env.sh
    spack arch -o
    spack arch -g

where ``$SPACKDIR`` is the directory where the Spack repository was cloned by
deployment.  This should hopefully give
you something close to what Spack wants.  If you get something like
``x86_64_v4`` for the target, use ``x86_64`` instead.

If you are getting other error messages, do your best to debug them but also
feel free to get in touch with the compass development team and we'll help if
we can.

If you get everything working well, please feel free to make a pull request
into the compass main repo to add your supported machine.
