.. _dev_quick_start:

Quick Start for Developers
==========================

.. _dev_shell:

Unix Shell
----------

Currently, compass only supports ``bash`` and related unix shells (such as
``ksh`` on the Mac).  We do not support ``csh``, ``tcsh`` or other variants of
``csh``.  An activation script for those shells will not be created.

If you normally use ``csh``, ``tcsh`` or similar, you will need to temporarily
switch to bash by calling ``/bin/bash`` each time you want to use compass.

.. _dev_compass_repo:

Set up a compass repository: for beginners
------------------------------------------

To begin, obtain the main branch of the
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
that can get you started.

You will need to run ``./conda/configure_compass_env.py`` each time you check
out a new branch or create a new worktree with ``git``.  Typically, you will
*not* need to run this command when you make changes to files within the
``compass`` python package.  These will automatically be recognized because
``compass`` is installed into the conda environment in "editable" mode.  You
*will* need to run the command if you add new code files or data files to the
package because these don't get added automatically.

Whether you are on one of the :ref:`dev_supported_machines` or an "unknown"
machine, you will need to specify a path where
`Miniforge3 <https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3>`_ either has
already been installed or an empty directory where the script can install it.
You must have write permission in the base environment (if it exists).

.. note::

    We have found that an existing Miniconda3 installation **does not** always
    work well for ``compass``, so please start with Miniforge3 instead.

.. note::

    It is *very* important that you not use a shared installation of Miniforge3
    or Miniconda3 such as the base environment for E3SM-Unified for ``compass``
    development. Most developers will not have write access to shared
    environments, meaning that you will get write-permission errors when you
    try to update the base environment or create the compass development
    environment.

    For anyone who does have write permission to a shared environment, you
    would be creating your compass development environment in a shared space,
    which could cause confusion.

    Please use your own personal installation of Miniforge3 for development,
    letting ``configure_compass_env.py`` download and install Miniforge3 for
    you if you don't already have it installed.

Supported machines
~~~~~~~~~~~~~~~~~~

If you are on one of the :ref:`dev_supported_machines`, run:

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <base_path_to_install_or_update_conda> \
        -c <compiler> [--mpi <mpi>] [-m <machine>] [--with_albany] \
        [--with_petsc]

The ``<base_path_to_install_or_update_conda>`` is typically ``~/miniforge3``.
This is the location where you would like to install Miniforge3 or where it is
already installed. If you have limited space in your home directory, you may
want to give another path.  If you already have it installed, that path will
be used to add (or update) the compass test environment.

See the machine under :ref:`dev_supported_machines` for a list of available
compilers to pass to ``-c``.  If you don't supply a compiler, you will get
the default one for that machine (usually Intel). Typically, you will want the
default MPI flavor that compass has defined for each compiler, so you should
not need to specify which MPI version to use but you may do so with ``--mpi``
if you need to.

If you are on a login node, the script should automatically recognize what
machine you are on.  You can supply the machine name with ``-m <machine>`` if
you run into trouble with the automatic recognition (e.g. if you're setting
up the environment on a compute node, which is not recommended).

Environments with Albany
~~~~~~~~~~~~~~~~~~~~~~~~

If you are working with MALI, you should specify ``--with_albany``.  This will
ensure that the Albany and Trilinos libraries are included among those built
with system compilers and MPI libraries, a requirement for many MAlI test
cases.  Currently, only Albany is only supported with ``gnu`` compilers.

It is safe to add the ``--with_albany`` flag for MPAS-Ocean but it is not
recommended unless a user wants to be able to run both models with the same
conda/spack environment.  The main downside is simply that unneeded libraries
will be linked in to MPAS-Ocean.

Environments with PETSc and Netlib-LAPACK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are working with MPAS-Ocean test cases that need PETSc and
Netlib-LAPACK, you should specify ``--with_petsc`` to
point to Spack environments where these libraries are included.  Appropriate
environment variables for pointing to these libraries will be build into the
resulting load script (see below).

Unknown machines
~~~~~~~~~~~~~~~~

If your are on an "unknown" machine, typically a Mac or Linux laptop or
workstation, you will need to specify which flavor of MPI you want to use
(``mpich`` or ``openmpi``):

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> --mpi <mpi>

Again, the ``<conda_path>`` is typically ``~/miniforge3``, and is the location
where you would like to install Miniforge3 or where it is already installed.
If you already have it installed, that path will be used to add (or update) the
compass test environment.

We only support one set of compilers for Mac and Linux (``gnu`` for Linux and
``clang`` with ``gfortran`` for Mac), so there is no need to specify them.
See :ref:`dev_other_machines` for more details.

In addition, unknown machines require a config file to be specified when setting
up the compass test environment.  A config file can be specified using
``-f <filename>``, where ``<filename>`` is an absolute or relative path to the
file. More information, including example config files, can be found
in :ref:`config_files`.

.. note::

    Currently, there is not a good way to build Albany for an unknown machine as
    part of the compass deployment process, meaning MALI will be limited to the
    shallow-ice approximation (SIA) solver.

    To get started on HPC systems that aren't supported by Compass, get in touch
    with the developers.

What the script does
~~~~~~~~~~~~~~~~~~~~

In addition to installing Miniforge3 and creating the conda environment for
you, this script will also:

* install `Jigsaw <https://github.com/dengwirda/jigsaw>`_ and
  `Jigsaw-Python <https://github.com/dengwirda/jigsaw-python>`_ from source
  from the `jigsaw-python` submodule. These tools are used to create many of
  the meshes used in Compass.

* install the ``compass`` package from the local branch in "development" mode
  so changes you make to the repo are immediately reflected in the conda
  environment.

* with the ``--update_spack`` flag on supported machines, installs or
  reinstalls a spack environment with various system libraries.  The
  ``--spack`` flag can be used to point to a location for the spack repo to be
  checked out.  Without this flag, a default location is used. Spack is used to
  build several libraries with system compilers and MPI library, including:
  `ParallelIO <https://github.com/NCAR/ParallelIO>`_ (parallel i/o in NetCDF
  format) `ESMF <https://earthsystemmodeling.org/>`_ (making mapping files
  in parallel), `Trilinos <https://trilinos.github.io/>`_,
  `Albany <https://github.com/sandialabs/Albany>`_,
  `Netlib-LAPACK <http://www.netlib.org/lapack/>`_ and
  `PETSc <https://petsc.org/>`_. **Please uses these flags with caution, as
  they can affect shared environments!**  See :ref:`dev_deploying_spack`.

* with the ``--with_albany`` flag, creates or uses an existing Spack
  environment that includes Albany and Trilinos.

* with the ``--with_petsc`` flag, creates or uses an
  existing Spack environment that includes PETSc and Netlib-LAPACK.

* make an activation script called ``load_*.sh``, where the details of the
  name encode the conda environment name, the machine, compilers, MPI
  libraries, and optional libraries,  e.g.
  ``load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh`` (``<version>``
  is the compass version, ``<machine>`` is the name of the
  machine, ``<compiler>`` is the compiler name, and ``mpi`` is the MPI flavor).

* optionally (with the ``--check`` flag), run some tests to make sure some of
  the expected packages are available.

Optional flags
~~~~~~~~~~~~~~

``--check``
    Check to make sure expected commands are present

``--python``
    Select a particular python version (the default is currently 3.13)

``--env_name``
    Set the name of the environment (and the prefix for the activation script)
    to something other than the default (``dev_compass_<version>`` or
    ``dev_compass_<version>_<mpi>``).

``--update_jigsaw``
    Used to reinstall Jigsaw and Jigsaw-Python into the conda environment if
    you have made changes to the Jigsaw (c++) code in the ``jigsaw-python``
    submodule. You should not need to reinstall Jigsaw-Python if you have made
    changes only to the python code in ``jigsaw-python``, as the python package
    is installed in
    `edit mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_.

Activating the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh

This will load the appropriate conda environment, load system modules for
compilers, MPI and libraries needed to build and run MPAS components, and
set environment variables needed for MPAS or ``compass``.  It will also set an
environment variable ``LOAD_COMPASS_ENV`` that points to the activation script.
``compass`` uses this to make an symlink to the activation script called
``load_compass_env.sh`` in the work directory.  When the load script is
executed from the base of the compass repository (i.e., as
``source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh``),
it will install the version of the ``compass`` package from that location into the associated
conda environment.  When the load script is executed from the work
directory through the symlink, it will activate the associated conda
environment, but does *not* install the ``compass`` package into the conda
environment; it is assumed that is already up to date from when the conda
environment was created or last updated.

It is generally recommended to activate the ``compass`` environment (from
either the compass repo or via the workdir symlink) from a
clean environment.  Unexpected behavior may occur if activating a different
``compass`` environment after having one already activated.

Once you have sourced the activation script, you can run ``compass`` commands
anywhere, and it always refers to that branch.  To find out which branch you
are actually running ``compass`` from, you should run:

.. code-block:: bash

    echo $LOAD_COMPASS_ENV

This will give you the path to the load script, which will also tell you where
the branch is.  If you do not use the worktree approach, you will also need to
check what branch you are currently on with ``git log``, ``git branch`` or
a similar command.

If you wish to work with another compiler, simply rerun the configure script with a new
compiler name and an activation script will be produced.  You can then source
either activation script to get the same conda environment but with different
compilers and related modules.  Make sure you are careful to set up compass by
pointing to a version of the MPAS model that was compiled with the correct
compiler.

Switching between different compass environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many developers are switching between different ``compass`` branches.
We have 2 main workflows for doing this: checking out different branches
in the same directory (with ``git checkout``) or creating new directories for
each branch (with ``git worktree``).  Either way, you need to be careful that
the version of the ``compass`` package that is installed in the conda
environment you are using is the one you want.  But how to handle it
differs slightly between these workflows.

If you are developing or using multiple ``compass`` branches in the same
directory (switching between them using ``git checkout``), you will need
to make sure you update your ``compass`` environment after changing
branches.  Often the branches you're developing will make use of the
same conda environment, because they are using the same
``compass`` version (so the dependencies aren't changing).  The same
conda environment (e.g. ``dev_compass_<version>``) can safely be used
with multiple branches if you explicitly reinstall the ``compass`` package
you want to use into the conda environment *after* moving to a new branch.
You can do this by simply re-executing
``source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh``
from the *root of the repo* before proceeding.

Similarly, if you are developing or using multiple ``compass`` branches
but you use a different directory for each
(creating the directories with ``git worktree``),
you will need to make sure the version of the ``compass`` package
in your conda environment is the one you want.
If your branches use the same ``compass`` version (so the dependencies
are the same), you can use the same conda environment
(e.g. ``dev_compass_<version>``) for all of them.  But you will only
be able to test one of them at a time.  You will tell the conda environment
which branch to use by running
``source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh``
from the *root of the directory (worktree) you want to work with* before
proceeding.

In both of these workflows, you can modify the ``compass`` code and the conda
environment will notice the changes as you make them.  However, if you have
added or removed any files during your development, you need to source the
load script again:
``source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh``
in the root of the repo or worktree so that the added or removed files will be
accounted for in the conda environment.

If you know that ``compass`` has different dependencies
in a branch or worktree you are working on compared to a previous branch
you have worked with (or if you aren't sure), it is safest to not just reinstall
the ``compass`` package but also to check the dependencies by re-running:
``./conda/configure_compass_env.py``  with the same arguments as above.
This will also reinstall the ``compass`` package from the current directory.
The activation script includes a check to see if the version of compass used
to produce the load script is the same as the version of compass in the
current branch.  If the two don't match, an error like the following results
and the environment is not activated:

.. code-block::

    $ source load_compass_test_morpheus_gnu_openmpi.sh
    This load script is for a different version of compass:
    __version__ = '1.2.0-alpha.6'

    Your code is version:
    __version__ = '1.2.0-alpha.7'

    You need to run ./conda/configure_compass_env.py to update your conda
    environment and load script.

If you need more than one conda environment (e.g. because you are testing
multiple branches at the same time), you can choose your own name
for the conda environment.  Typically, this might be something related to the
name of the branch you are developing.  This can be done with the
``--env-name`` argument to ``./conda/configure_compass_env.py``.  You
can reuse the same custom-named environment across multiple branches
if that is useful.  Just remember to reinstall ``compass`` each time you
switch branches.

.. note::

    If you switch branches and *do not* remember to recreate the conda
    environment (``./conda/configure_compass_env.py``) or at least source the
    activation script (``load_dev_compass*.sh``), you are likely to end up with
    an incorrect and possibly unusable ``compass`` package in your conda
    environment.

    In general, if one wishes to switch between environments created for
    different compass branches or applications, the best practice is to end
    the current terminal session and start a new session with a clean
    environment before executing the other compass load script.  Similarly,
    if you want to run a job script that itself sources the load script,
    it's best to start a new terminal without having sourced a load script at
    all.

.. note::

    With the conda environment activated, you can switch branches and update
    just the ``compass`` package with:

    .. code-block:: bash

        python -m pip install --no-deps --no-build-isolation -e .

    The activation script will do this automatically when you source it in
    the root directory of your compass branch.  The activation script will also
    check if the current compass version matches the one used to create the
    activation script, thus catching situations where the dependencies are out
    of date and the configure script needs to be rerun.

Troubleshooting
~~~~~~~~~~~~~~~

If you run into trouble with the environment or just want a clean start, you
can run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c <compiler> --recreate

The ``--recreate`` flag will delete the conda environment and create it from
scratch.  This takes just a little extra time.

.. _dev_creating_only_env:

Creating/updating only the compass environment
----------------------------------------------

For some workflows (e.g. for MALI development with the Albany library when the
MALI build environment has been created outside of ``compass``, for example,
on an unsupported machine), you may only want to create the conda environment
and not build PIO, ESMF or other packages with system compilers, or
include any system modules or environment variables in your activation script.
In such cases, run with the ``--env_only`` flag:

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <conda_path> --env_only

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    source ./load_dev_compass_<version>.sh

This will load the appropriate conda environment for ``compass``.  It will also
set an environment variable ``LOAD_COMPASS_ENV`` that points to the activation
script. ``compass`` uses this to make a symlink to the activation script
called ``load_compass_env.sh`` in the work directory.

If you switch to another branch, you will need to rerun:

.. code-block:: bash

    ./conda/configure_compass_env.py --conda <conda_path> --env_only

to make sure dependencies are up to date and the ``compass`` package points
to the current directory.

.. note::

    With the conda environment activated, you can switch branches and update
    just the ``compass`` package with:

    .. code-block:: bash

        python -m pip install --no-deps --no-build-isolation -e .

    This will be substantially faster than rerunning
    ``./conda/configure_compass_env.py ...`` but at the risk that dependencies are
    not up-to-date.  Since dependencies change fairly rarely, this will usually
    be safe.


.. _dev_build_mpas:

Building MPAS components
------------------------

The MPAS repository is a submodule of the compass repository.  For example, to
compile MPAS-Ocean:

.. code-block:: bash

    source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh
    cd E3SM-Project/components/mpas-ocean/
    make <mpas_make_target>

MALI can be compiled with or without the Albany library that contains the
first-order velocity solver.  The Albany first-order velocity solver is the
only velocity option that is scientifically validated, but the Albany library
is not available for every compiler yet.  Therefore, in some situations
it is desirable to compile without Albany to run basic tests on platforms where
Albany is not available.  This basic mode of MALI can be compiled similarly to
MPAS-Ocean, i.e.:

.. code-block:: bash

    source ./load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh
    cd MALI-Dev/components/mpas-albany-landice
    make <mpas_make_target>

Compiling MALI with Albany has not yet been standardized.  Some information is
available at
`https://github.com/MALI-Dev/E3SM/wiki <https://github.com/MALI-Dev/E3SM/wiki>`_,
and complete instructions will be added here in the future.

See the last column of the table in :ref:`dev_supported_machines` for the right
``<mpas_make_target>`` command for each machine and compiler.


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

where ``$MACHINE`` is an ES3M machine, ``$WORKDIR`` is the location where compass
test cases will be set up and ``$MPAS`` is the directory where the MPAS model
executable has been compiled. See :ref:`dev_compass_setup` for details.

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
(``load_dev_compass_<version>_<machine>_<compiler>_<mpi>.sh``) that you used to set
up the suite or test case (``load_compass_env.sh`` is just a symlink to that
activation script you sourced before setting up the suite or test case).

.. _dev_compass_style:

Code style for compass
----------------------

``compass`` complies with the coding conventions of
`PEP8 <https://peps.python.org/pep-0008/>`_. Rather than memorize all the
guidelines, the easiest way to stay in compliance as a developer writing new
code or modifying existing code is to use a PEP8 style checker. One option is
to use an IDE with a PEP8 style checker built in, such as
`PyCharm <https://www.jetbrains.com/pycharm/>`_. See
`this tutorial <https://www.jetbrains.com/help/pycharm/tutorial-code-quality-assistance-tips-and-tricks.html>`_
for some tips on checking code style in PyCharm.


Here's the manual way to check for PEP8 compliance.

`Flake8 <https://flake8.pycqa.org/en/latest/>`_ is a PEP8 checker that is
included in the ``compass`` conda environment. For each of the files you have
modified, you can run the Flake8 checker to see a list of all instances of
non-compliance in that file.

.. code-block:: bash

    $flake8 example.py
    example.py:77:1: E302 expected 2 blank lines, found 1

For this example, we would just add an additional blank line after line 77 and
run the checker again to make sure we've resolved the issue.

Once you open a pull request for your feature, there is an additional PEP8
style checker at this stage.

.. _dev_compass_repo_advanced:

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
    git clone git@github.com:MPAS-Dev/compass.git main
    cd main

The ``MPAS-Dev/compass`` repo is now ``origin``. You can add more remotes. For
example:

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
unix directory:

.. code-block:: bash

    cd compass/main
    git worktree add -b new_branch_name ../new_branch_name origin/main
    cd ../new_branch_name

In this example, we branched off ``origin/main``, but you could start from
any branch, which is specified by the last ``git worktree`` argument.

There are two ways to build the MPAS executable:

1. Compass submodule (easier): This guarantees that the MPAS commit matches
   compass.  It is also the default location for finding the MPAS model so you
   don't need to specify the ``-p`` flag at the command line or put the MPAS
   model path in your config file (if you even need a config file at all):

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
   is compatible with the version of ``compass`` that you are using.  The
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
     git checkout -b your_new_branch origin/main
