.. _dev_machine_anvil:

Anvil
=====

intel
-----

This is the default ``compass`` compiler on Anvil.  If the environment has
been set up properly (see :ref:`dev_conda_env`), you should be able to source:

.. code-block:: bash

    source load_dev_compass_1.0.0_anvil_intel_impi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] intel-mpi

For other MPI libraries (``openmpi`` or ``mvapich`` instead of ``impi``), use

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] ifort

gnu
---

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source load_dev_compass_1.0.0_anvil_gnu_openmpi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gfortran
