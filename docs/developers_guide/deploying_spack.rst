.. _dev_deploying_spack:

*********************************
Deploying a new spack environment
*********************************

Where do we update compass dependencies?
========================================

Mache
-----

If system modules change in E3SM, we try to stay in sync:

* compilers

* MPI libraries

* netcdf-C

* netcdf-fortran

* pnetcdf

* mkl (or other linear algebra libs)

When we update the mache version in compass, we also need to bump the compass
version (typically either the major or the minor version) and then re-deploy
shared spack environments on each supported machine.

Spack
-----

Spack is for libraries used by MPAS and tools that need system MPI:

* ESMF

* SCORPIO

* Albany

* PETSc

* Netlib LAPACK

When we update the versions of any of these libraries in compass, we also need
to bump the compass version (typically either the major or the minor version)
and then re-deploy shared spack environments on each supported machine.

Conda
-----

Conda (via conda-forge) is used for python packages and related dependencies
that don’t need system MPI. Conda environments aren’t shared between
developers because the compass you’re developing is part of the conda
environment.

When we update the constraints on conda dependencies, we also need to bump the
compass alpha, beta or rc version.  We do not need to re-deploy spack
environments on share machines because they remain unaffected.

Mache
=====

A brief tour of mache.

Identifying E3SM machines
-------------------------

Compass and other packages use mache to identify what machine they’re on (when
you’re on a login node).  This is used when configuring compass and creating a
conda environment.  Because of this, there is a “bootstrapping” process where
a conda environment is created with mache installed in it, then the machine is
identified and the rest of the compass setup can continue.

Config options describing E3SM machines
---------------------------------------

Mache has config files for each E3SM machine that tell us where E3SM-Unified
is, where to find diagnostics, and much more, see
`machines <https://github.com/E3SM-Project/mache/tree/main/mache/machines>`_.
These config options are shared across packages including:

* MPAS-Analysis

* E3SM_Diags

* zppy

* compass

* E3SM-Unified

Compass uses these config options to know how to make a job script, where to
find locally cached files in its “databases” and much more.

Modules, env. variables, etc. for  E3SM machines
------------------------------------------------

Mache keeps its own copy of the E3SM file
`config_machines.xml <https://github.com/E3SM-Project/E3SM/blob/master/cime_config/machines/config_machines.xml>`_
in the package `here <https://github.com/E3SM-Project/mache/blob/main/mache/cime_machine_config/config_machines.xml>`_.
We try to keep a close eye on E3SM master and update mache when system modules
for machines that mache knows about get updated.  When this happens, we update
mache’s copy of ``config_machines.xml`` and that tells me which modules to
update in spack, see below.dev_quick_start

Mirroring MOAB on Chicoma
-------------------------

The firewall on LANL IC's Chicoma blocks access to the MOAB package (at least
at the moment -- Xylar has made a request to allow access).  To get around
this, someone testing or deploying spack builds on Chicoma will first need to
update the local spack mirror with the desired version of MOAB (5.5.1 in this
example).

First, you need to know the versions of the ``mache`` and ``moab`` packages
that are needed (1.20.0 and 5.5.1, respectively, in this example).  These are
specified in ``conda/configure_compass_env.py`` and ``conda/default.cfg``,
respectively.  On a LANL laptop with either (1) the VPN turned off and the
proxies unset or (2) the VPN turned on and the proxies set, run:

.. code-block:: bash

    MACHE_VER=1.20.0
    MOAB_VER=5.5.1
    mkdir spack_mirror
    cd spack_mirror
    git clone git@github.com:E3SM-Project/spack.git -b spack_for_mache_${MACHE_VER} spack_for_mache_${MACHE_VER}
    source spack_for_mache_${MACHE_VER}/share/spack/setup-env.sh

    # remove any cache files that might cause trouble
    rm -rf ~/.spack

    # this should create spack_mirror with subdirectories moab and _source-cache
    spack mirror create -d spack_mirror moab@${MOAB_VER}

    tar cvfj spack_mirror.tar.bz2 spack_mirror

Then, if you used option (1) above turn on the LANL VPN (and set the proxies).
You may find it convenient to login on to Chicoma
(e.g. ``ssh -tt wtrw 'ssh ch-fe'``) in a separate terminal if you have
configured your laptop to preserve connections.

.. code-block:: bash

    rsync -rLpt -e 'ssh wtrw ssh' spack_mirror.tar.bz2 ch-fe:/usr/projects/e3sm/compass/chicoma-cpu/spack/


Then, on Chicoma:

.. code-block:: bash

    cd /usr/projects/e3sm/compass/chicoma-cpu/spack/
    tar xvf spack_mirror.tar.bz2
    chgrp -R climate spack_mirror/
    chmod -R ug+w spack_mirror/
    chmod -R ugo+rX spack_mirror/
    rm spack_mirror.tar.bz2

Creating spack environments
---------------------------

Mache has templates for making spack environments on some of the E3SM supported
machines.  See `spack <https://github.com/E3SM-Project/mache/tree/main/mache/spack>`_.
It also has functions for building the spack environments with these templates
using E3SM’s fork of spack (see below).


Updating spack from compass with mache from a remote branch
===========================================================

If you haven’t cloned compass and added my fork, here’s the process:

.. code-block:: bash

    mkdir compass
    cd compass/
    git clone git@github.com:MPAS-Dev/compass.git main
    cd main/
    git remote add xylar/compass git@github.com:xylar/compass.git

Now, we need to set up compass and build spack packages making use of the
updated mache.  This involves changing the mache version in a couple of places
in compass and updating the version of compass itself to a new alpha, beta or
rc.  As an example, we will use the branch
`simplify_local_mache <https://github.com/xylar/compass/tree/simplify_local_mache>`_.

Often, we will need to test with a ``mache`` branch that has changes needed
by compass.  Here, we will use ``<fork>`` as a stand-in for the fork of mache
to use (e.g. ``E3SM-Project/mache``) and ``<branch>`` as the stand-in for a branch on
that fork (e.g. ``main``).

We also need to make sure there is a spack branch for the version of compass.
The spack branch is a branch off of the develop branch on
`E3SM’s spack repo <https://github.com/E3SM-Project/spack>`_ that has any
updates to packages required for this version of mache.  The remote branch
is named after the release version of mache (omitting any alpha, beta or rc
suffix because it is intended to be the spack branch we will use once the
``mache`` release happens).  In this example, we will work with the branch
`spack_for_mache_1.12.0 <https://github.com/E3SM-Project/spack/tree/spack_for_mache_1.12.0>`_.
The local clone is instead named after the compass version (again any omitting
alpha, beta or rc) plus the compiler and MPI library because we have discovered
two users cannot make modifications to the same git clone.  Giving each clone
of the spack branch a unique name ensures that they are independent.

Here's how to get a branch of compass we're testing (``simplify_local_mache``
in this case) as a local worktree:

.. code-block:: bash

    # get my branch
    git fetch --all -p
    # make a worktree for checking out my branch
    git worktree add ../simplify_local_mache -b simplify_local_mache \
        --checkout xylar/compass/simplify_local_mache
    cd ../simplify_local_mache/

You will also need a local installation of
`Miniforge <https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3>`_.
Compass can do this for you if you haven't already installed it.  If you want
to download it manually, use the Linux x86_64 version for all our supported
machines.

.. note::

    We have found that an existing Miniconda3 installation **does not** always
    work well for compass, so please start with Miniforge3 instead.

.. note::

  You definitely need your own local Miniforge3 installation -- you can’t use
  a system version or a shared one like E3SM-Unified.

Define a location where Miniforge3 is installed or where you want to install
it:

.. code-block:: bash

    # change to your conda installation
    export CONDA_BASE=${HOME}/miniforge

Okay, we're finally ready to do a test spack build for compass.
To do this, we call the ``configure_compass_env.py`` script using
``--mache_fork``, ``--mache_branch``, ``--update_spack``, ``--spack`` and
``--tmpdir``. Here is an example appropriate for Anvil or Chrysalis:

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --mache_fork <fork> \
        --mache_branch <branch> \
        --update_spack \
        --spack /lcrc/group/e3sm/${USER}/spack_test \
        --tmpdir ${TMPDIR} \
        --compiler intel intel gnu \
        --mpi openmpi impi openmpi \
        --recreate

The directory you point to with ``--conda`` either doesn't exist or contains
your existing installation of Miniforge3.

When you supply ``--mache_fork`` and ``--mache_branch``, compass will clone
a fork of the ``mache`` repo and check out the requested branch, then install
that version of mache into both the compass installation conda environment and
the final compass environment.

``mache`` gets installed twice because the deployment tools need ``mache`` to
even know how to install compass and build the spack environment on supported
machines.  The "prebootstrap" step in deployment is creating the installation
conda environment.  The "bootstrap" step is creating the conda environment that
compass will actually use and (in this case with ``--update_spack``) building
spack packages, then creating the "load" or "activation" script that you will
need to build MPAS components and run compass.

For testing, you want to point to a different location for installing spack
using ``--spack``.

On many machines, the ``/tmp`` directory is not a safe place to build spack
packages.  Use ``--tmpdir`` to point to another place, e.g., your scratch
space.

The ``--recreate`` flag may not be strictly necessary but it’s a good idea.
This will make sure both the bootstrapping conda environment (the one that
installs mache to identify the machine) and the compass conda environment are
created fresh.

The ``--compiler`` flag is a list of one or more compilers to build for and the
``--mpi`` flag is the corresponding list of MPI libraries.  To see what is
supported on each machine, take a look at :ref:`dev_supported_machines`.

Be aware that not all compilers and MPI libraries support Albany and PETSc, as
discussed below.

Testing spack with PETSc (and Netlib LAPACK)
--------------------------------------------

If you want to build PETSc (and Netlib LAPACK), use the ``--with_petsc`` flag.
Currently, this only works with some
compilers, but that may be more that I was trying to limit the amount of work
for the compass support team.  There is a file,
`petsc_supported.txt <https://github.com/MPAS-Dev/compass/blob/main/conda/petsc_supported.txt>`_,
that lists supported compilers and MPI libraries on each machine.

Here is an example:

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --mache_fork <fork> \
        --mache_branch <branch> \
        --update_spack \
        --spack /lcrc/group/e3sm/${USER}/spack_test \
        --tmpdir ${TMPDIR} \
        --compiler intel gnu \
        --mpi openmpi \
        --with_petsc \
        --recreate \
        --verbose

Testing spack with Albany
-------------------------

If you also want to build Albany, use the ``--with_albany`` flag.  Currently,
this only works with Gnu compilers.  There is a file,
`albany_support.txt <https://github.com/MPAS-Dev/compass/blob/main/conda/albany_supported.txt>`_,
that lists supported compilers and MPI libraries on each machine.

Here is an example:

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --mache_fork <fork> \
        --mache_branch <branch> \
        --update_spack \
        --spack /lcrc/group/e3sm/${USER}/spack_test \
        --tmpdir ${TMPDIR} \
        --compiler gnu \
        --mpi openmpi \
        --with_albany \
        --recreate

Troubleshooting spack
---------------------

If you encounter an error like:
.. code-block:: none

    ==>   spack env activate dev_compass_1_2_0-alpha_6_gnu_mpich
    ==> Error: Package 'armpl' not found.
    You may need to run 'spack clean -m'.

during the attempt to build spack, you will first need to find the path to
``setup-env.sh`` (see ``compass/build_*/build*.sh``) and source that script to
get the ``spack`` command, e.g.:

.. code-block:: bash

    source ${PSCRATCH}/spack_test/dev_compass_1_2_0-alpha_6_gnu_mpich/share/spack/setup-env.sh

Then run the suggested command:

.. code-block:: bash

    spack clean -m

After that, re-running ``./conda/configure_compass_env.py`` should work correctly.

This issue seems to be related to switching between spack v0.18 and v0.19 (used by different versions of compass).

Testing compass
===============

Testing MPAS-Ocean without PETSc
--------------------------------

Please use the E3SM-Project submodule in compass for testing, rather than
E3SM’s master branch.  The submodule is the version we know works with compass
and serves as kind of a baseline for other testing.

.. code-block:: bash

    # source whichever load script is appropriate
    source load_dev_compass_1.2.0-alpha.5_chrysalis_intel_openmpi.sh
    git submodule update --init --recursive
    cd E3SM-Project/components/mpas-ocean
    # this will build with PIO and OpenMP
    make ifort
    compass suite -s -c ocean -t pr -p . \
        -w /lcrc/group/e3sm/ac.xylar/compass/test_20230202/ocean_pr_chrys_intel_openmpi
    cd /lcrc/group/e3sm/ac.xylar/compass/test_20230202/ocean_pr_chrys_intel_openmpi
    sbatch job_script.pr.bash

You can make other worktrees of E3SM-Project for testing other compilers if
that’s helpful.  It also might be good to open a fresh terminal to source a
new load script.  This isn’t required but you’ll get some warnings.

.. code-block:: bash

    source load_dev_compass_1.2.0-alpha.5_chrysalis_gnu_openmpi.sh
    cd E3SM-Project
    git worktree add ../e3sm_chrys_gnu_openmpi
    cd ../e3sm_chrys_gnu_openmpi
    git submodule update --init --recursive
    cd components/mpas-ocean
    make gfortran
    compass suite -s -c ocean -t pr -p . \
        -w /lcrc/group/e3sm/ac.xylar/compass/test_20230202/ocean_pr_chrys_gnu_openmpi
    cd /lcrc/group/e3sm/ac.xylar/compass/test_20230202/ocean_pr_chrys_gnu_openmpi
    sbatch job_script.pr.bash

You can also explore the utility in
`utils/matrix <https://github.com/MPAS-Dev/compass/tree/main/utils/matrix>`_ to
test on several compilers automatically.

Testing MALI with Albany
------------------------

Please use the MALI-Dev submodule in compass for testing, rather than MALI-Dev
develop branch.  The submodule is the version we know works with compass and
serves as kind of a baseline for other testing.

.. code-block:: bash

    # source whichever load script is appropriate
    source load_dev_compass_1.2.0-alpha.5_chrysalis_gnu_openmpi_albany.sh
    git submodule update --init --recursive
    cd MALI-Dev/components/mpas-albany-landice
    # you need to tell it to build with Albany
    make ALBANY=true gfortran
    compass suite -s -c landice -t full_integration -p . \
        -w /lcrc/group/e3sm/ac.xylar/compass/test_20230202/landice_full_chrys_gnu_openmpi
    cd /lcrc/group/e3sm/ac.xylar/compass/test_20230202/landice_full_chrys_gnu_openmpi
    sbatch job_script.full_integration.bash

Testing MPAS-Ocean with PETSc
-----------------------------

The tests for PETSc use nonhydrostatic capabilities not yet integrated into
E3SM.  So you can’t use the E3SM-Project submodule.  You need to use Sara
Calandrini’s `nonhydro <https://github.com/scalandr/E3SM/tree/ocean/nonhydro>`_
branch.

.. code-block:: bash

    # source whichever load script is appropriate
    source load_dev_compass_1.2.0-alpha.5_chrysalis_intel_openmpi_petsc.sh
    git submodule update --init
    cd E3SM-Project
    git remote add scalandr/E3SM git@github.com:scalandr/E3SM.git
    git worktree add ../nonhydro_chrys_intel_openmpi -b nonhydro_chrys_intel_openmpi \
        --checkout scalandr/E3SM/ocean/nonhydro
    cd ../nonhydro_chrys_intel_openmpi
    git submodule update --init --recursive
    cd components/mpas-ocean
    # this will build with PIO, Netlib LAPACK and PETSc
    make ifort
    compass list | grep nonhydro
    # update these numbers for the 2 nonhydro test cases
    compass setup -n 245 246 -p . \
        -w /lcrc/group/e3sm/ac.xylar/compass/test_20230202/nonhydro_chrys_intel_openmpi
    cd /lcrc/group/e3sm/ac.xylar/compass/test_20230202/nonhydro_chrys_intel_openmpi
    sbatch job_script.custom.bash

As with non-PETSc MPAS-Ocean and MALI, you can have different worktrees with
Sara’s nonhydro branch for building with different compilers or use
`utils/matrix <https://github.com/MPAS-Dev/compass/tree/main/utils/matrix>`_ to
build (and run).

Deploying shared spack environments
===================================

.. note::

  Be careful about deploying shared spack environments, as changes you make
  can affect other compass users.

Once compass has been tested with the spack builds in a temporary location, it
is time to deploy the shared spack environments for all developers to use.
A ``mache`` developer will make a ``mache`` release (if needed) before this
step begins.  So there is no need to build mache from a remote branch anymore.

Compass knows where to deploy spack on each machine because of the ``spack``
config option specified in the ``[deploy]`` section of each machine's config
file, see the `machine configs <https://github.com/MPAS-Dev/compass/tree/main/compass/machines>`_.

It is best to update the remote compass branch in case of changes:

.. code-block:: bash

    cd simplify_local_mache
    # get any changes
    git fetch --all -p
    # hard reset if there are changes
    git reset –hard xylar/compass/simplify_local_mache

Deploy spack for compass without Albany or PETSc
------------------------------------------------

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --update_spack \
        --tmpdir ${TMPDIR} \
        --compiler intel intel gnu \
        --mpi openmpi impi openmpi \
        --recreate

Deploying spack with Albany
---------------------------

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --update_spack \
        --tmpdir ${TMPDIR} \
        --compiler gnu \
        --mpi openmpi \
        --with_albany \
        --recreate

Deploying spack with PETSc (and Netlib LAPACK)
----------------------------------------------

.. code-block:: bash

    export TMPDIR=/lcrc/group/e3sm/${USER}/spack_temp
    ./conda/configure_compass_env.py \
        --conda ${CONDA_BASE} \
        --update_spack \
        --tmpdir ${TMPDIR} \
        --compiler intel gnu \
        --mpi openmpi \
        --with_petsc \
        --recreate \
        --verbose
