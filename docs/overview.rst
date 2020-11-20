.. _quick_start:

Quick Start
===========

Set up a COMPASS repository: for beginners
------------------------------------------

To begin, obtain the master branch of the
`compass repository <https://github.com/MPAS-Dev/compass>`_ with:

.. code-block:: bash

    git clone git@github.com:MPAS-Dev/compass.git
    cd compass
    git submodule update --init --recursive

The MPAS repository is a submodule of COMPASS repository.  For example, to
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


.. _conda_env:

compass conda environment
-------------------------

The compass conda environment includes all the python libraries needed to run
compass scripts. It is maintained on our standard machines (Grizzly, Badger,
Anvil, Compy and Cori).  Here are the commands for each machine:

* grizzly and badger:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/load_latest_compass.sh

* anvil (blues):

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

* compy:

.. code-block:: bash

    source /share/apps/E3SM/conda_envs/load_latest_compass.sh

* cori:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/anaconda_envs/load_latest_compass.sh

To install your own compass conda environment on other machines, first, install
`Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ (if miniconda is
not already installed), then create a new conda environment as follows:

.. code-block:: bash

    conda create -n compass -c conda-forge -c e3sm python=3.8 compass

Each time you want to work with COMPASS, you will need to run:

.. code-block:: bash

    conda activate compass

.. _setup_overview:

Setting up a test case
----------------------

To see all available test cases you can set up in compass, starting in the base
of your local compass repo or branch, run:

.. code-block:: bash

    ./list_testcases.py

and you get output like this:

.. code-block:: none

    69: -o ocean -c global_ocean -r QU240 -t init
    70: -o ocean -c global_ocean -r QU240 -t performance_test

To set up a particular test case, you can either use the full sequence of flags:

.. code-block:: bash

    ./setup_testcase.py \
      --config_file config.ocean \
      --work_dir $WORKDIR \
      --model_runtime runtime_definitions/mpirun.xml \
      -o ocean -c global_ocean -r QU240 -t init

or you can replace the last line with the simple shortcut: ``-n 69``.

Here ``$WORKDIR`` is a path, usually to your scratch space. For example,

.. code-block:: bash

    --work_dir /lustre/scratch4/turquoise/$USER/runs/191210_test_new_branch

and ``config.ocean`` is a config file that specifies directory and
file paths. You can make a copy of the template config file for your core
(e.g. ``cp general.config.ocean config.ocean``) and modify it with the
appropriate paths to the appropriate MPAS-Model build and local caches for
meshes and initial-condition data files.  The documentation for
:ref:`setup_ocean` includes some examples you can use as a starting point for
specific machines. (Similar documentation for the ``landice`` core will is
coming soon.)

The ``--model_runtime`` is either ``srun`` or ``mpirun``, depending whether your
machine uses the SLURM queuing system or not.


Running a test case
-------------------

After compiling the code and setting up a test case, you can log into an
interactive node (see machine instructions below) and then

.. code-block:: bash

    cd $WORKDIR
    cd ocean/global_ocean/QU240/init
    ./run.py

Note the sequence of subdirectories is the same as the flags used to set up the
case.

In order to run a bit-for-bit test with a previous case, use
``-b $PREVIOUS_WORKDIR``.


Regression suites
-----------------

We have assembles suites of test cases for code regressions and bit-for-bit
testing. For the ocean core, they are here:

.. code-block:: bash

    ls testing_and_setup/compass/ocean/regression_suites/
       land_ice_fluxes.xml  light.xml  nightly.xml  rpe_tests.xml

You can set up a regression as follows:

.. code-block:: bash

    ./manage_regression_suite.py -s \
       --config_file config.ocean \
       -t ocean/regression_suites/nightly.xml \
       --model_runtime runtime_definitions/mpirun.xml \
       --work_dir $WORKDIR

where the details are mostly the same as for setting up a case. You can use the
same ``config.ocean`` file and use ``-b $PREVIOUS_WORKDIR`` for bit-for-bit
comparison of the output with a previous nightly regression suite.

To run the regression suite, log into an interactive node, load your modules,
and

.. code-block:: bash

    cd $WORKDIR
    ./nightly_ocean_test_suite.py


Set up a COMPASS repository with worktrees: for advanced users
--------------------------------------------------------------

This section uses git worktree, which provides more flexibility but is more
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
    git worktree add -b newBranchName ../newBranchName origin/master
    cd ../newBranchName

In this example, we branched off ``origin/master``, but you could start from
any branch, which is specified by the last ``git worktree`` argument.

In each new branch directory that you make, you will need to make a copy of
``general.config.ocean`` or ``general.config.landice`` and alter the copy to
point to the MPAS executable and files. There are two ways to point to the MPAS
executable:

1. Compass submodule (easier): This guarantees that the MPAS commit matches
   compass.

   .. code-block:: bash


     git submodule update --init --recursive
     cd MPAS-Model/ocean/develop/
     # load modules (see machine-specific instructions below)
     make gfortran CORE=ocean

2. Other MPAS directory (advanced): Create your own MPAS-Model repository
   elsewhere on disk, make a copy of ``general.config.ocean`` or
   ``general.config.landice``, and point the copy to the MPAS-Model paths.
   The user must ensure that flag names and test cases match appropriately.
   The simplest way to set up a new MPAS repo in a new directory is:

   .. code-block:: bash

     git clone git@github.com:MPAS-Dev/MPAS.git your_new_branch
     cd your_new_branch
     git checkout -b your_new_branch origin/ocean/develop

Note that for ocean development, it is best to branch from ``ocean/develop`` and
for MALI development, start with ``landice/develop``.
