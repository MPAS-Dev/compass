.. _dev_machine_anvil:

Anvil
=====

intel18
-------

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m anvil -c intel18

To build the MPAS model with

.. code-block:: bash

    make CORE=landice ifort

or

.. code-block:: bash

    make CORE=ocean ifort

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
