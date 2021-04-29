.. _dev_machine_anvil:

Anvil
=====

intel18
-------

This is the default ``compass`` compiler on Anvil.  To activate the compass
environment, load modules, and set appropriate environment variables, run this
in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh

To build the MPAS model with

.. code-block:: bash

    make CORE=landice ifort

or

.. code-block:: bash

    make CORE=ocean ifort

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
