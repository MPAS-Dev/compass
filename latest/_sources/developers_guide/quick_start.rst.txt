.. _dev_quick_start:

Quick Start for Developers
==========================

.. _dev_compass_repo:

Set up a compass repository: for beginners
------------------------------------------

To begin, obtain the master branch of the
`compass repository <https://github.com/MPAS-Dev/compass>`_ with:

.. code-block:: bash

    git clone git@github.com:MPAS-Dev/compass.git
    cd compass
    git submodule update --init --recursive

The E3SM repository and a clone of E3SM for MALI development are submodules of
the compass repository.

.. _dev_conda_env:

compass conda environment, compilers and system modules
-------------------------------------------------------

As a developer, you will need your own environment with the latest dependencies
for compass and a development installation of ``compass`` from the branch
you're working on.

The ``conda`` directory in the repository has a tool ``configure_compass_env.py``
that can get you started.  If you are on one of the :ref:`dev_supported_machines`,
run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c <compiler>

If you don't have `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_
installed in ``<conda_path>``, it will be downloaded and installed for you in
this location. If you already have it installed, that path will be used to add
(or update) the compass test environment.

See the machine under :ref:`dev_supported_machines` for a list of available
compilers to pass to ``-c``.  If you don't supply a compiler, you will get
the default one for that machine (usually Intel). Typically, you will want the
default MPI flavor that compass has defined for each compiler, so you should
not need to specify which MPI version to use but you may do so with ``-i`` if
you need to.

If you are on a login node, the script should automatically recognize what
machine you are on.  You can supply the machine name with ``-m <machine>`` if
you run into trouble with the automatic recognition (e.g. if you're setting
up the environment on a compute node, which is not recommended).

In addition to installing Miniconda and creating the conda environment for you,
this script will also:

* install the ``compass`` package from the local branch in "development" mode
  so changes you make to the repo are immediately reflected in the conda
  environment.

* build the `SCORPIO <https://github.com/E3SM-Project/scorpio>`_ library if it
  hasn't already been built.  SCORPIO is needed building and running MPAS
  components.

* build the `ESMF <https://earthsystemmodeling.org/>`_ library if it hasn't
  already been built.  ESMF with the system's version of MPI is needed for
  making mapping files.

* make an activation script called
  ``test_compass_<version>_<machine>_<compiler>_<mpi>.sh``,
  where ``<version>`` is the compass version, ``<machine>`` is the name of the
  machine (to prevent confusion when running from the same branch on multiple
  machines), ``<compiler>`` is the compiler name (e.g. ``intel`` or ``gnu``),
  and ``mpi`` is the MPI flavor (e.g. ``impi``, ``mvapich``, ``openmpi``).

* optionally (with the ``--check`` flag), run some tests to make sure some of
  the expected packages are available.

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    source ./test_compass_<version>_<machine>_<compiler>_<mpi>.sh

This will load the appropriate conda environment, load system modules for
compilers, MPI and libraries needed to build and run MPAS components, and
set environment variables needed for MPAS or ``compass``.  It will also set an
environment variable ``LOAD_COMPASS_ENV`` that points to the activation script.
``compass`` uses this to make an symlink to the activation script called
``load_compass_env.sh`` in the work directory.

If you switch to another branch, you need to rerun

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <conda_path> -c <compiler>

to make sure dependencies are up to date and the ``compass`` package points
to the current directory.

.. note::

    With the conda environment activated, you can switch branches and update
    just the ``compass`` package with:

    .. code-block:: bash

        python -m pip install -e .

    This will be substantially faster than rerunning
    ``./conda/configure_compass_env.py ...`` but at the risk that dependencies are
    not up-to-date.  Since dependencies change fairly rarely, this will usually
    be safe.

If you wish to work with another compiler, simply rerun the script with a new
compiler name and an activation script will be produced. You can then source
either activation script to get the same conda environment but with different
compilers and related modules.  Make sure you are careful to set up compass by
pointing to a version of the MPAS model that was compiled with the correct
compiler.

If you run into trouble with the environment or just want a clean start, you
can run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c <compiler> --recreate

The ``--recreate`` flag will delete the conda environment and create it from
scratch.  This takes just a little extra time.

You can check to make sure expected commands are present with ``--check``, you
can select a particular python version with ``--python``, you can set the name
of the environment (and the prefix for the activation script) something other
than the default (``test_compass<version>``) with ``--env-name``.

If you are not on a supported machine, you need to choose your MPI type
(``mpich`` or ``openmpi``) with the ``--mpi`` flag.  The compilers are
automatically ``gnu`` for Linux and ``clang`` (with ``gfortran``) for OSX, so
you do not need to specify those.

.. _dev_creating_only_env:

Creating/updating only the compass environment
----------------------------------------------

For some workflows (e.g. for MALI development wih the Albany library), you may
only want to create the conda environment and not build SCORPIO, ESMF or
include any system modules or environment variables in your activation script.
In such cases, run with the ``--env_only`` flag.

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <conda_path> --env_only

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    source ./test_compass_<version>.sh

This will load the appropriate conda environment for ``compass``.  It will also
set an environment variable ``LOAD_COMPASS_ENV`` that points to the activation
script. ``compass`` uses this to make an symlink to the activation script
called ``load_compass_env.sh`` in the work directory.

If you switch to another branch, you need to rerun

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <conda_path> --env_only

to make sure dependencies are up to date and the ``compass`` package points
to the current directory.

.. note::

    With the conda environment activated, you can switch branches and update
    just the ``compass`` package with:

    .. code-block:: bash

        python -m pip install -e .

    This will be substantially faster than rerunning
    ``./conda/configure_compass_env.py ...`` but at the risk that dependencies are
    not up-to-date.  Since dependencies change fairly rarely, this will usually
    be safe.

.. _dev_working_with_compass:

Running compass from the repo
-----------------------------

If you follow the procedure above, you can run compass with the ``compass``
command-line tool exactly like described in the User's Guide :ref:`quick_start`
and as detailed in :ref:`dev_command_line`.

To list test cases you need to run:

.. code-block:: bash

    compass list

The results will be the same as described in :ref:`setup_overview`, but the
test cases will come from the local ``compass`` directory.

To set up a test case, you will run something like:

.. code-block:: bash

    compass setup -t ocean/global_ocean/QU240/mesh -m $MACHINE -w $WORKDIR -p $MPAS

To list available test suites, you would run:

.. code-block:: bash

    compass list --suites

And you would set up a suite as follows:

.. code-block:: bash

    compass suite -s -c ocean -t nightly -m $MACHINE -w $WORKDIR -p $MPAS

When you want to run the code, go to the work directory (for the suite or test
case), log onto a compute node (if on an HPC machine) and run:

.. code-block:: bash

    source load_compass_env.sh
    compass run

The first command will source the same activation script
(``test_compass_<version>_<machine>_<compiler>_<mpi>.sh``) that you used to set
up the suite or test case (``load_compass_env.sh`` is just a symlink to that
activation script you sourced before setting up the suite or test case).

Building MPAS components
------------------------

The MPAS repository is a submodule of compass repository.  For example, to
compile MPAS-Ocean:

.. code-block:: bash

    source ./test_compass_<version>_<machine>_<compiler>_<mpi>.sh
    cd E3SM-Project/components/mpas-ocean/
    make <mpas_compiler>

For MALI:

.. code-block:: bash

    source ./test_compass_<version>_<machine>_<compiler>_<mpi>.sh
    cd MALI-Dev/components/mpas-albany-landice
    make <mpas_compiler>

See :ref:`dev_supported_machines` for the right ``<mpas_compiler>`` command for
each machine and compiler.


Set up a compass repository with worktrees: for advanced users
--------------------------------------------------------------

This section uses ``git worktree``, which provides more flexibility but is more
complicated. See the beginner section above for the simpler version. In the
worktree version, you will have many unix directories, and each corresponds to
a git branch. It is easier to keep track of, and easier to work with many
branches at once. Begin where you keep your repositories:

.. code-block:: bash

    mkdir compass
    cd compass
    git clone git@github.com:MPAS-Dev/compass.git master
    cd master

The ``MPAS-Dev/compass`` repo is now ``origin``. You can add more remotes. For
example

.. code-block:: bash

    git remote add mark-petersen git@github.com:mark-petersen/compass.git
    git fetch mark-petersen

To view all your remotes:

.. code-block:: bash

    git remote -v

To view all available branches, both local and remote:

.. code-block:: bash

    git branch -a

We will use the git worktree command to create a new local branch in its own
unix directory.

.. code-block:: bash

    cd compass/master
    git worktree add -b new_branch_name ../new_branch_name origin/master
    cd ../new_branch_name

In this example, we branched off ``origin/master``, but you could start from
any branch, which is specified by the last ``git worktree`` argument.

There are two ways to build the MPAS executable:

1. Compass submodule (easier): This guarantees that the MPAS commit matches
   compass.  It is also the default location for finding the MPAS model so you
   don't need to specify the ``-p`` flag at the command line or put the MPAS
   model path in your config file (if you even need a config file at all).

   .. code-block:: bash

     git submodule update --init --recursive
     cd E3SM-Project/components/mpas-ocean/
     # load modules
     make gfortran

   For the "load modules" step, see :ref:`machines` for specific instructions.

2. Other E3SM directory (advanced): Create your own clone of the
   ``E3SM-Project/E3SM`` or ``MALI-Dev/E3SM`` repository elsewhere on disk.
   Either make an ``ocean.cfg`` or ``landice.cfg`` that specifies the absolute
   path to the path where the ``ocean_model`` or ``landice_model`` executable
   is found, or specify this path on the command line with ``-p``.  You are
   responsible for knowing if this particular version of MPAS component's code
   is compatible with the version of ``compass`` that you are using. The
   simplest way to set up a new repo for MALI development in a new directory
   is:

   .. code-block:: bash

     git clone git@github.com:MALI-Dev/E3SM.git your_new_branch
     cd your_new_branch
     git checkout -b your_new_branch origin/develop


   The equivalent for MPAS-Ocean development would be:

   .. code-block:: bash

     git clone git@github.com:E3SM-Project/E3SM.git your_new_branch
     cd your_new_branch
     git checkout -b your_new_branch origin/master
