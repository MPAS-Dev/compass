.. _dev_machine_grizzly:

Grizzly
=======

.. note::

  It is important that you not point to custom module files for the compiler,
  MPI, and netcdf modules on Grizzly to work properly.  If you have:

  .. code-block:: bash

    module use /usr/projects/climate/SHARED_CLIMATE/modulefiles/all

  or similar commands in your ``.bashrc``, please either comment them out or
  make sure to run

  .. code-block:: bash

    module unuse /usr/projects/climate/SHARED_CLIMATE/modulefiles/all


intel
-----

This is the default ``compass`` compiler on Grizzly.  If the environment has
been set up properly (see :ref:`dev_conda_env`), you should be able to source:

.. code-block:: bash

    source load_dev_compass_1.0.0_brizzly_intel_impi.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make intel-mpi

gnu
---

If you've set things up for this compiler, you should be able to:

.. code-block:: bash

    source load_dev_compass_1.0.0_grizzly_gnu_mvapich.sh

Then, you can build the MPAS model with

.. code-block:: bash

    make gfortran
