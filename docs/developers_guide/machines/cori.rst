Cori
====

cori-haswell, intel
-------------------

This is the default ``compass`` architecture and compiler on Cori.  If the
environment has been set up properly (see :ref:`dev_conda_env`), you should be
able to source:

.. code-block:: bash

    source test_compass_1.0.0_cori-haswell_intel_mpt.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make CORE=landice intel-nersc

or

.. code-block:: bash

    make CORE=ocean intel-nersc

cori-haswell, gnu
-----------------

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source test_compass_1.0.0_cori-haswell_gnu_mpt.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc

cori-knl, intel
---------------

This is the default ``compass`` compiler on Cori-KNL.  If the environment has
been set up properly (see :ref:`dev_conda_env`), you should be able to source:

.. code-block:: bash

    source test_compass_1.0.0_cori-knl_intel_mpt.sh

Then, you can build the MPAS model with


.. code-block:: bash

    make CORE=landice intel-mpi

or

.. code-block:: bash

    make CORE=ocean intel-mpi

cori-knl, gnu
-------------

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source test_compass_1.0.0_cori-knl_gnu_mpt.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make CORE=landice gnu-nersc

or

.. code-block:: bash

    make CORE=ocean gnu-nersc
