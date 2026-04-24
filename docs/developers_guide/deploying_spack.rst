.. _dev_deploying_spack:

*********************************
Deploying a new spack environment
*********************************

Compass now deploys environments through ``./deploy.py``, backed by
``mache.deploy``.  Compared with pixi dependency updates, Spack updates are
heavier-weight because the resulting environments are often shared on
supported machines and build times can be substantial.

Use this page when you need to update or redeploy the shared Spack-backed
Compass stack on supported HPC systems.

What lives where?
=================

The main deployment inputs in this repository are:

``deploy.py``
    The repository entry point for deployment.

``deploy/pins.cfg``
    Version pins for pixi, Spack and shared dependencies.

``deploy/config.yaml.j2``
    Deployment behavior and runtime settings consumed by ``mache.deploy``.

``deploy/spack.yaml.j2``
    The Compass-specific list of Spack specs to deploy.

``deploy/hooks.py``
    Compass-specific deployment hooks for machine defaults and optional Albany
    support.

Machine-specific Spack environment templates now live in
`mache <https://github.com/E3SM-Project/mache>`_, not in this repository.

When do shared Spack environments need to be updated?
=====================================================

Shared Spack environments typically need to be redeployed when one of the
following changes:

* the pinned versions in ``deploy/pins.cfg`` under ``[spack]`` or ``[all]``

* the requested specs in ``deploy/spack.yaml.j2``

* machine-specific Spack behavior in ``mache`` that affects supported systems

Because these environments are shared, it is usually appropriate to bump the
Compass version before redeploying them.

The current automated Compass Spack deployment covers the packages listed in
``deploy/spack.yaml.j2``.  Compass now splits these into per-toolchain
library environments for MPAS link dependencies such as SCORPIO and optional
Albany/Trilinos, plus a shared software environment for tool binaries such as
ESMF, MOAB and CMake.  PETSc and Netlib-LAPACK are no longer part of the
automated ``./deploy.py`` workflow in this repository.

Testing a Spack deployment
==========================

Before touching any shared location, do a test deployment in a scratch or
user-owned Spack path.

Basic test deployment
---------------------

On a supported machine, run something like:

.. code-block:: bash

    SCRATCH_SPACK=<path_to_test_spack>
    ./deploy.py --deploy-spack --spack-path ${SCRATCH_SPACK} \
        --compiler intel intel gnu --mpi openmpi impi openmpi --recreate

Adjust ``--compiler`` and ``--mpi`` to match the machine you are testing.

Useful flags
------------

``--deploy-spack``
    Build or update the supported Spack library and software environments.

``--spack-path <path>``
    Use a test Spack checkout and install location instead of the shared one.

``--compiler <compiler...>`` and ``--mpi <mpi...>``
    Select the toolchain combinations to test.

``--recreate``
    Start from a clean deployment for this run.

``--with-albany``
    Deploy the Albany-enabled Spack library environment in addition to the
    default one.

If you need a non-default temporary directory for Spack builds, set
``spack.tmpdir`` in ``deploy/config.yaml.j2`` before running deployment.

Testing against a mache branch
------------------------------

When Compass changes depend on unreleased ``mache`` updates, test with a fork
and branch explicitly:

.. code-block:: bash

    SCRATCH_SPACK=<path_to_test_spack>
    ./deploy.py --mache-fork <fork> --mache-branch <branch> \
        --deploy-spack --spack-path ${SCRATCH_SPACK} \
        --compiler intel intel gnu --mpi openmpi impi openmpi --recreate

This keeps the Compass deployment workflow the same while swapping in the
requested ``mache`` branch during bootstrap and deployment.

Albany deployments
------------------

If you are testing MALI workflows that require Albany, add
``--with-albany``:

.. code-block:: bash

    SCRATCH_SPACK=<path_to_test_spack>
    ./deploy.py --deploy-spack --spack-path ${SCRATCH_SPACK} \
        --compiler gnu --mpi openmpi --with-albany --recreate

Albany support is restricted to the machine/compiler/MPI combinations allowed
by the deployment hooks and machine configuration.

Validating the deployment
=========================

After deployment completes:

* source the generated load script for the toolchain you want to validate

* build the appropriate MPAS component with the matching make target

* set up and run the relevant Compass suites or test cases

* confirm the generated work directories contain ``load_compass_env.sh`` and
  that sourcing it activates the intended environment

The :ref:`dev_quick_start` page and the machine pages under
:ref:`dev_supported_machines` remain the main references for build targets and
day-to-day usage once deployment has succeeded.

Troubleshooting Spack deployment
================================

If a Spack deployment fails part way through and suggests clearing caches, the
usual recovery step is:

.. code-block:: bash

    source <spack_path>/share/spack/setup-env.sh
    spack clean -m

Then rerun ``./deploy.py`` with the same arguments.

If deployment is failing because the wrong architecture or operating system is
being inferred for a new machine template, use:

.. code-block:: bash

    source <spack_path>/share/spack/setup-env.sh
    spack arch -o
    spack arch -g

and update the corresponding machine support in ``mache``.

If a machine requires a local mirror or site-specific workaround for one of the
Spack packages, coordinate that change with the machine maintainers and keep
the Compass pins in ``deploy/pins.cfg`` aligned with whatever is available at
the site.

Deploying shared Spack environments
===================================

.. note::

    Be careful about deploying shared Spack environments, because changes you
    make can affect other Compass users on that machine.

Once test deployments and validation passes are complete, deploy to the shared
location by omitting ``--spack-path`` and using the machine defaults from the
``[deploy]`` section of the relevant machine config:

.. code-block:: bash

    ./deploy.py --deploy-spack \
        --compiler intel intel gnu --mpi openmpi impi openmpi --recreate

For an Albany-enabled shared deployment, run:

.. code-block:: bash

    ./deploy.py --deploy-spack \
        --compiler gnu --mpi openmpi --with-albany --recreate

Do this only after the equivalent test deployments have succeeded in a private
Spack path.
