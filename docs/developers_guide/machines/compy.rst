.. _machine_compy:

CompyMcNodeFace
===============

So far, we have not got ``compass`` to successfully build and run Compy.  We
will update this page when we do.

intel
-----

This works to build (but not yet run) standalone MPAS.  Again, we will update
as soon as we have a solution.

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m compy -c intel

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

