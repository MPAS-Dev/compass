.. _dev_machine_compy:

CompyMcNodeFace
===============

.. note::

    So far, we have not got ``compass`` to successfully build and run Compy.
    We will update this page when we do. See
    `this issue <https://github.com/MPAS-Dev/compass/issues/57>`_ for more
    discussion.

intel
-----

This works to build (but not yet run) standalone MPAS.  Again, we will update
as soon as we have a solution.

This is the default ``compass`` compiler on CompyMcNodeFace.  To activate the
compass environment, load modules, and set appropriate environment variables,
run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

