Cori
====

cori-haswell, intel
-------------------

This is the default ``compass`` architecture and compiler on Cori.  If the
environment has been set up properly (see :ref:`dev_conda_env`), you should be
able to source:

.. code-block:: bash

    source load_dev_compass_1.0.0_cori-haswell_intel_mpt.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] intel-cray

cori-haswell, gnu
-----------------

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source load_dev_compass_1.0.0_cori-haswell_gnu_mpt.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gnu-cray
