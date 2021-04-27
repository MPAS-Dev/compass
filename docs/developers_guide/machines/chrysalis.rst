.. _dev_machine_chrysalis:

Chrysalis
=========

intel
-----

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m chrysalis -c intel

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

gnu
---

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m anvil -c gnu

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran
