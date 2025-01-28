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

If you follow the procedure in :ref:`dev_conda_env`, you will have an
activation script for activating the development conda environment, setting
loading system modules and setting environment variables so you can build
MPAS and work with ``compass``.  Just source the script that should appear in
the base of your compass branch, e.g.:

.. code-block:: bash

    source load_dev_compass_1.0.0_anvil_intel_impi.sh

After loading this environment, you can set up test cases or test suites, and
a link ``load_compass_env.sh`` will be included in each suite or test case
work directory.  This is a link to the activation script that you sourced when
you were setting things up.  You can can source this file on a compute node
(e.g. in a job script) to get the right compass conda environment, compilers,
MPI libraries and environment variables for running ``compass`` tests and
the MPAS model.

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

If you are working on an "unknown" machine, the procedure is pretty similar
to what was described in :ref:`dev_conda_env`.  The main difference is that
we will use ``mpich`` or ``openmpi`` and the gnu compilers from conda-forge
rather than system compilers.  To create a development conda environment and
an activation script for it, on Linux, run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c gnu -i mpich

and on OSX run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c clang -i mpich

You may use ``openmpi`` instead of ``mpich`` but we have had better experiences
with the latter.

The result should be an activation script ``load_dev_compass_1.0.0_<mpi>.sh``.
Source this script to get the appropriate conda environment and environment
variables.

Under Linux, you can build the MPAS model with

.. code-block:: bash

    make gfortran

Under OSX, you can build the MPAS model with

.. code-block:: bash

    make gfortran-clang

.. _dev_add_supported_machine:

Adding a New Supported Machine
------------------------------

If you want to add a new supported machine, you need to add both a config file
and a yaml file describing your machine, as detailed below.

Adding a Machine Config File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step in adding a new supported machine to add a config file in
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

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /home/xylar/data/mpas/compass_envs


    # Options related to deploying a compass conda environment on supported
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
``compass_envs`` as a path where shared conda environments will be installed
for compass releases.  If developers always create their own conda
environments, this path will never be used.

In ``[deploy]``, you will specify config options used in setting up conda
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
``--machine <machine>`` each time they setup compass environments or test
cases.


Describing a Spack Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to create a template YAML file that can be used to create
`Spack environments <https://spack-tutorial.readthedocs.io/en/latest/tutorial_environments.html>`_
for your machine. Compass uses Spack environments to build packages that need
MPI support or which should be build for some other reason with system
compilers rather than coming from pre-built conda packages. Using a Spack
environment allows these packages to be built together in a consistent way that
is not guaranteed if you try to install dependencies one-by-one.  In Spack
parlance, this is known as
`unified concretization <https://spack.readthedocs.io/en/latest/environments.html#spec-concretization>`_.

To do this, you will create a file ``conda/spack/<machine>_<compiler>_<mpi>.yaml``
similar to the following example for an Ubuntu laptop:

.. code-block::

    spack:
      specs:
      - gcc
      - openmpi
    {{ specs }}
      concretizer:
        unify: true
      packages:
        all:
          compiler: [gcc@11.3.0]
          providers:
            mpi: [openmpi]
        curl:
          externals:
          - spec:  curl@7.81.0
            prefix: /usr
          buildable: false
        gcc:
          externals:
          - spec: gcc@11.3.0
            prefix: /usr
          buildable: false
      config:
        install_missing_compilers: false
      compilers:
      - compiler:
          spec: gcc@11.3.0
          paths:
            cc: /usr/bin/gcc
            cxx: /usr/bin/g++
            f77: /usr/bin/gfortran
            fc: /usr/bin/gfortran
          flags: {}
          operating_system: ubuntu22.04
          target: x86_64
          modules: []
          environment: {}
          extra_rpaths: []


Typically your system will already have compilers if nothing else, and this is
what we assume here.  Give the appropriate path (replace ``/usr`` with the
appropriate path on your system).  We have had better luck with ``gcc`` than
other compilers like Intel so far so for new supported machines so that's our
recommendation.  Use ``gcc --version`` to determine the version and replace
``11.3.0`` with this number.

Finally, you might need to update the ``target`` and ``operating_system``.
This is a bit of a "catch 22" in that you can use Spack to find this out but
compass is designed to clone and set up Spack for you so we assume you don't
have it yet.  For now, make your best guess using the info on
`this page <https://spack.readthedocs.io/en/latest/basic_usage.html#architecture-specifiers>`_
and correct it later if necessary.

You may need to load a system module to get the compilers and potentially other
libraries such as MPI, HDF5, and NetCDF-C if you prefer to use system modules
rather than having Spack build them.  If this is the case, the best way to do
this is to add a file
``conda/spack/<machine>_<compiler>_<mpi>.sh`` along these lines:

.. code-block:: bash

    module purge
    module load perl/5.32.0-bsnc6lt
    module load gcc/9.2.0-ugetvbp
    module load openmpi/4.1.3-sxfyy4k
    module load intel-mkl/2020.4.304-n3b5fye
    module load hdf5/1.10.7-j3zxncu
    module load netcdf-c/4.4.1-7ohuiwq
    module load netcdf-fortran/4.4.4-k2zu3y5
    module load parallel-netcdf/1.11.0-mirrcz7

These modules will be loaded either before or after the spack environment,
depending on the ``modules_before`` and ``modules_after`` config options above.
You can also add modules in your YAML file but this shouldn't be necessary.

For examples from various supported machines, compilers and MPI libraries, see the
`mache spack directory <https://github.com/E3SM-Project/mache/tree/main/mache/spack>`_.

Building the Spack Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to try setting up compass and asking it to build the Spack
environment with a command something like:

.. code-block:: bash

  ./conda/configure_compass_env.py --verbose --update_spack --conda <conda_path> -c gnu -i openmpi ...

The ``--update_spack`` flag tells compass to create (or update) a Spack
environment.  You can specify a directory for testing Spack with the
``--spack`` flag.  You can specify a temporary directory for building spack
packages with ``--tmpdir`` (this directory must already exist).  This is useful
if your ``/tmp`` space is small (Spack will use several GB of temporary space).


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

where ``$SPACKDIR`` is the directory where the Spack repository was cloned
by compass (you should see ``Cloning into <$SPACKDIR>`` in the terminal, which
will hopefully help you find the right directory).  This should hopefully give
you something close to what Spack wants.  If you get something like
``x86_64_v4`` for the target, use ``x86_64`` instead.

If you are getting other error messages, do your best to debug them but also
feel free to get in touch with the compass development team and we'll help if
we can.

If you get everything working well, please feel free to make a pull request
into the compass main repo to add your supported machine.
