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

The MPAS repository is a submodule of compass repository.  For example, to
compile MPAS-Ocean:

.. code-block:: bash

    cd MPAS-Model/ocean/develop/
    # load modules (see machine-specific instructions below)
    make gfortran CORE=ocean

For MALI:

.. code-block:: bash

    cd MPAS-Model/landice/develop/
    # load modules (see machine-specific instructions below)
    make gfortran CORE=landice


.. _dev_conda_env:

compass conda environment
-------------------------

As a developer, you will nearly always want to set up your own conda
environment with the latest dependencies for compass.  You can try using the
latest compass environment on a supported machine or making one from the latest
release (see :ref:`conda_env`) but there is a risk that the dependencies will
not be correct and things will not work as expected as a result.

To install your own compass conda environment on other machines, first, install
`Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ (if miniconda is
not already installed), then add the
`conda-forge channel <https://conda-forge.org/#about>`_:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict

then create a new conda environment (called ``dev_compass`` in this example) as
follows:

.. code-block:: bash

    conda create -n dev_compass python=3.8 affine cartopy cartopy_offlinedata \
        cmocean "esmf=*=mpi_mpich*" ffmpeg "geometric_features=0.4.0" git \
        ipython "jigsaw=0.9.14" "jigsawpy=0.3.3" jupyter lxml matplotlib \
        metis "mpas_tools=0.5.1" mpich nco "netcdf4=*=nompi_*" numpy \
        progressbar2 pyamg "pyremap>=0.0.7,<0.1.0" rasterio requests scipy \
        xarray

We will do our best to keep this list of dependencies in sync with the
"official" list, which is found in
`recipe/meta.yaml <https://github.com/MPAS-Dev/compass/blob/master/recipe/meta.yaml>`_

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    conda activate dev_compass

.. _dev_working_with_compass:

Running compass from the repo
-----------------------------

If you are working with the released ``compass`` package, you can interact with
it directly with the ``compass`` command-line tool as described in
:ref:`setup_overview` and :ref:`suite_overview`.  If you are developing the
code out of a repository, though, you need to call the local code with:

.. code-block:: bash

    python -m compass ...

This way, you will use the code in the local ``compass`` directory.  If you
are running out of a ``dev_compass`` environment like described above, you
won't have a ``compass`` command-line tool to run.  If you are using one of
the release environments (e.g. because dependencies haven't changed since the
last release), you want to be careful not to run the ``compass`` command-line
tool directly because you won't be accessing the code you're working on.

To list test cases you need to run:
.. code-block:: bash

    python -m compass list

The results will be the same as described in :ref:`setup_overview`, but the
test cases will come from the local ``compass`` directory.

To set up a test case, you will run something like:

.. code-block:: bash

    python -m compass setup -t ocean/global_ocean/QU240/mesh -m $MACHINE -w $WORKDIR -p $MPAS

To list available test suites, you would run:

.. code-block:: bash

    python -m compass list --suites

And you would set up a suite as follows:

.. code-block:: bash

    python -m compass suite -s -c ocean -t nightly -m $MACHINE -w $WORKDIR -p $MPAS

Otherwise, things are the same as in :ref:`suite_overview`.

You will see symlinks to the ``compass`` package in the base work directory
for suites as well as each test case and step's work directory.  These are to
make sure that the code from the repository is also used when you run test
cases and steps.  You can even use the symlinks as a convenient way to access
and edit the code as you're testing your changes.

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
     cd MPAS-Model/ocean/develop/
     # load modules
     make gfortran CORE=ocean

   For the "load modules" step, see :ref:`machines` for specific instructions.

2. Other MPAS directory (advanced): Create your own clone of the MPAS-Model
   repository elsewhere on disk.  Either make an ``ocean.cfg`` or
   ``landice.cfg`` that specifies the absolute path to MPAS-Model repo where
   the ``ocean_model`` or ``landice_model`` executable is found, or specify
   this path on the command line with ``-p``.  You are responsible for knowing
   if this particular version of MPAS-Model is compatible with the version of
   ``compass`` that you are using. The simplest way to set up a new MPAS repo
   in a new directory is:

   .. code-block:: bash

     git clone git@github.com:MPAS-Dev/MPAS.git your_new_branch
     cd your_new_branch
     git checkout -b your_new_branch origin/ocean/develop

Note that for ocean development, it is best to branch from ``ocean/develop``
and for MALI development, start with ``landice/develop``.
