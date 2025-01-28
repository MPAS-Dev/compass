.. _dev_machine_compy:

CompyMcNodeFace
===============

intel
-----

This works to build (but not yet run) standalone MPAS.  Again, we will update
as soon as we have a solution.

This is the default ``compass`` compiler on CompyMcNodeFace.  If the
environment has been set up properly (see :ref:`dev_conda_env`), you should be
able to source:

.. code-block:: bash

    source load_dev_compass_1.0.0_compy_intel_impi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] intel-mpi

