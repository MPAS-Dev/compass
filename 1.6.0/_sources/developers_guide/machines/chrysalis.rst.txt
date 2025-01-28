.. _dev_machine_chrysalis:

Chrysalis
=========

intel
-----

This is the default ``compass`` compiler on Chrysalis.  If the environment has
been set up properly (see :ref:`dev_conda_env`), you should be able to source:

.. code-block:: bash

    source load_dev_compass_1.0.0_chrysalis_intel_openmpi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] ifort

gnu
---

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source load_dev_compass_1.0.0_chrysalis_gnu_openmpi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gfortran
