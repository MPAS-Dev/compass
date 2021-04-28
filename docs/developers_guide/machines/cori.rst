Cori
====

cori-haswell, intel
-------------------

This is the default ``compass`` architecture and compiler on Cori.  To activate
the compass environment, load modules, and set appropriate environment
variables, run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-haswell

You don't have to supply ``-m cori-haswell`` but it will prevent you from
getting a warning that you might want ``cori-knl``.

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-nersc

or

.. code-block:: bash

    make CORE=ocean intel-nersc

cori-haswell, gnu
-----------------

To activate the compass environment, load modules, and set appropriate
environment variables, run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-haswell -c gnu

As above, you don't have to supply ``-m cori-haswell`` but it will prevent you
from getting a warning that you might want ``cori-knl``.

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc

cori-knl, intel
---------------

This is the default ``compass`` compiler on Cori-KNL.  To activate the compass
environment, load modules, and set appropriate environment variables, run this
in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-knl

To build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-nersc

or

.. code-block:: bash

    make CORE=ocean intel-nersc

cori-knl, gnu
-------------

To activate the compass environment, load modules, and set appropriate
environment variables, run this in the ``compass`` repo root:

.. code-block:: bash

    source ./load/load_compass_env.sh -m cori-knl -c gnu


To build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc

