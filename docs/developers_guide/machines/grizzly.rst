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
