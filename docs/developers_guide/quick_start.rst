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

compass pixi and Spack environments, compilers and system modules
-----------------------------------------------------------------

As a developer, you will need your own deployment environment with the latest
dependencies for compass and a development installation of ``compass`` from
the branch you're working on.

Compass now uses ``mache.deploy`` for deployment.  In this repository, the
entry point is ``./deploy.py``.

You will typically rerun ``./deploy.py`` each time you check out a new branch
or create a new worktree with ``git``.  In most cases, you do not need to
rerun deployment while you are editing existing files in the ``compass``
package because ``compass`` is installed in editable mode.

.. note::

    ``./deploy.py`` expects ``pixi`` to be available either on ``PATH`` or at
    ``~/.pixi/bin/pixi``.  If your ``pixi`` executable lives somewhere else,
    pass it explicitly with ``--pixi <path>``.

Supported machines
~~~~~~~~~~~~~~~~~~

If you are on one of the :ref:`dev_supported_machines`, run:

.. code-block:: bash

    ./deploy.py [--machine <machine>] [--compiler <compiler> ...] \
        [--mpi <mpi> ...] [--deploy-spack] [--no-spack] \
        [--pixi-path <path>] [--recreate] [--with-albany]

If you are on a login node, machine detection typically works automatically.
You can pass ``--machine <machine>`` explicitly if needed.

By default, Compass will reuse existing machine-specific Spack environments
when the current deployment needs them.  On supported machines, this now means
per-toolchain library environments together with a shared software
environment for tool binaries such as ESMF and MOAB.  Use ``--deploy-spack``
when you want to build or update those Spack environments.  Use ``--no-spack``
for a pixi-only deployment.

Environments with Albany
~~~~~~~~~~~~~~~~~~~~~~~~

If you are working with MALI, use ``--with-albany`` so the Albany and
Trilinos libraries are included in the deployed Spack library environment.
Albany is currently only supported for some machine/compiler/MPI
combinations, most commonly ``gnu`` builds on supported machines.

Unknown machines
~~~~~~~~~~~~~~~~

If a machine is not known to ``mache``, add machine support first
(see :ref:`dev_add_supported_machine`).

For workflows that need custom machine config files, see :ref:`config_files`.

What the script does
~~~~~~~~~~~~~~~~~~~~

``./deploy.py`` can:

* create or update a local pixi deployment prefix (``pixi-env`` by default)

* install `Jigsaw <https://github.com/dengwirda/jigsaw>`_ and
  `Jigsaw-Python <https://github.com/dengwirda/jigsaw-python>`_ from the
  ``jigsaw-python`` submodule when needed

* install the ``compass`` package from the local branch in editable mode so
  changes you make to the repo are reflected immediately

* optionally deploy or reuse Spack library environments for selected
  compiler/MPI toolchains, plus a shared Spack software environment for
  supporting binaries

* generate activation scripts (``load_*.sh``)

Useful flags
~~~~~~~~~~~~

``--machine``
    Set the machine explicitly instead of relying on automatic detection

``--pixi-path``
    Choose the directory for the pixi environment (``pixi-env`` in the
    repository root by default).  ``--prefix`` is a deprecated alias.

``--compiler``, ``--mpi``
    Select compiler/MPI combinations, primarily for Spack deployment

``--deploy-spack``
    Deploy supported Spack library/software environments instead of only
    reusing existing ones

``--no-spack``
    Disable all Spack use for this run and rely on pixi dependencies instead

``--spack-path``
    Set the Spack checkout path used for deployment

``--recreate``
    Recreate deployment artifacts if they already exist

``--bootstrap-only``
    Update only the bootstrap pixi environment used internally by deployment

``--mache-fork``, ``--mache-branch``, ``--mache-version``
    Test deployment against a specific mache fork, branch or version

Activating the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time you want to work with compass, source one of the generated load
scripts:

.. code-block:: bash

    source ./load_*.sh

On supported machines, the scripts are named
``load_compass_<machine>_<compiler>_<mpi>.sh``, one for each compiler and MPI
combination you deployed.  Sourcing a load script checks that the ``compass``
package in the pixi environment is the version the script was generated for,
activates the pixi environment, loads machine modules and Spack environments
when appropriate, and sets environment variables needed by ``compass`` and
MPAS components.

When you are working inside a suite or test-case work directory, source
``load_compass_env.sh`` instead.  This is a symlink to the load script you
used while setting up the work directory.

The active load script path is exported in ``COMPASS_LOAD_SCRIPT``.  Compass
still accepts ``LOAD_COMPASS_ENV`` as a legacy fallback while the migration is
in progress.

If you wish to work with another compiler or MPI library, rerun
``./deploy.py`` with the desired options so the corresponding load script is
generated or refreshed.  Make sure you build MPAS with the same compiler and
MPI combination as the load script you plan to use.

Switching between different compass environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``./deploy.py`` installs the ``compass`` package in editable mode from the
directory where you run it.  Changes you make to existing files, and files you
add or remove, are picked up without redeploying or re-sourcing the load
script.  The pixi environment and load scripts do have to be updated when the
``compass`` version or its dependencies change.

Many developers switch between ``compass`` branches, either by checking out
different branches in the same directory (with ``git checkout``) or by
creating a directory for each branch (with ``git worktree``).

If you switch branches in the same directory, the editable install follows the
checkout, so there is nothing to do as long as the ``compass`` version and
dependencies are the same.  If the version has changed, the load script
refuses to activate the environment with an error like:

.. code-block:: none

    $ source load_compass_chrysalis_gnu_openmpi.sh
    verifying deployed compass version...
    ERROR: version mismatch for compass.
      Deployed: 2.0.0-alpha.2
      Runtime:  2.0.0-alpha.3

Rerun ``./deploy.py`` to update the environment and load scripts.  If you know
or suspect that the dependencies have changed, rerun ``./deploy.py`` even if
the version has not.

If you use a directory for each branch, the simplest approach is to run
``./deploy.py`` in each worktree.  By default, each worktree gets its own
``pixi-env`` directory and load scripts, so the worktrees are independent of
one another.

You can instead share one pixi environment between worktrees with
``./deploy.py --pixi-path <path>`` as long as the branches have the same
dependencies.  In that case, the ``compass`` package in the shared environment
points to whichever worktree you ran ``./deploy.py`` from most recently, and
sourcing a load script from another worktree does not change that.  To switch
worktrees, either rerun ``./deploy.py --pixi-path <path>`` from the worktree
you want to use or, with the environment activated, run:

.. code-block:: bash

    python -m pip install --no-deps --no-build-isolation -e .

from the root of that worktree.  The ``pip`` command is much faster than
``./deploy.py`` but skips the check that the dependencies are up to date.

.. note::

    If you wish to switch between environments created for different compass
    branches or applications, the best practice is to end the current terminal
    session and start a new session with a clean environment before sourcing
    the other compass load script.  Similarly, if you want to run a job script
    that itself sources the load script, it's best to start a new terminal
    without having sourced a load script at all.

Troubleshooting
~~~~~~~~~~~~~~~

If you run into trouble with the environment or just want a clean start, you
can run:

.. code-block:: bash

    ./deploy.py [--machine <machine>] [--compiler <compiler> ...] \
        [--mpi <mpi> ...] [--deploy-spack] [--no-spack] --recreate

The ``--recreate`` flag will delete the deployment artifacts and create them
from scratch.  This takes just a little extra time.

.. _dev_creating_only_env:

Creating/updating only the compass environment
----------------------------------------------

For some workflows, you may only want to create the pixi environment and not
build or reuse Spack environments.  In such cases, run:

.. code-block:: bash

    ./deploy.py --no-spack

When ``--no-spack`` is not used, omitting ``--deploy-spack`` still means
Compass will try to reuse any required pre-existing Spack environments.

To update only the bootstrap environment used internally by deployment, run:

.. code-block:: bash

    ./deploy.py --bootstrap-only

As above, source a generated load script each time you want to work with
compass.  The load script sets ``COMPASS_LOAD_SCRIPT``, which ``compass`` uses
to make the ``load_compass_env.sh`` symlink in each work directory.


.. _dev_build_mpas:

Building MPAS components
------------------------

The MPAS repository is a submodule of the compass repository.  For example, to
compile MPAS-Ocean:

.. code-block:: bash

    source ./load_*.sh
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

    source ./load_*.sh
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

The first command will source the same activation script that you used to set
up the suite or test case (``load_compass_env.sh`` is just a symlink to the
load script you sourced before setting up the suite or test case).

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
included in the ``compass`` development environment. For each of the files you
have modified, you can run the Flake8 checker to see a list of all instances
of non-compliance in that file.

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
