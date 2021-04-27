Cori
====

cori-haswell, gnu
-----------------

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-haswell -c gnu

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc


cori-haswell, intel
-------------------

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-haswell -c intel


To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-nersc

or

.. code-block:: bash

    make CORE=ocean intel-nersc

cori-knl, gnu
-------------

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-knl -c gnu


To build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc


cori-knl, intel
---------------

To load the compass environment and modules, and set appropriate environment
variables run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-knl -c intel

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-nersc

or

.. code-block:: bash

    make CORE=ocean intel-nersc
