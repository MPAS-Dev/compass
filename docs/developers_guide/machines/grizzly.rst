.. _dev_machine_grizzly:

Grizzly
=======

intel
-----

This is the default ``compass`` compiler on Grizzly.  To activate the compass
environment, load modules, and set appropriate environment variables, run this
in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

gnu
---

To activate the compass environment, load modules, and set appropriate
environment variables, run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -c gnu

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran
