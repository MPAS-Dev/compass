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

This is the default ``compass`` compiler on CompyMcNodeFace.  If the
environment has been set up properly (see :ref:`dev_conda_env`), you should be
able to source:

.. code-block:: bash

    source test_compass_1.0.0_compy_intel_impi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

