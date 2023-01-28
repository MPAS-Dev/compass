.. _dev_machines:

Machines
========

One of the major advantages of ``compass`` over :ref:`legacy_compass` is that it
attempts to be aware of the capabilities of the machine it is running on.  This
is a particular advantage for so-called "supported" machines with a config file
defined for them in the ``compass`` package.  But even for "unknown" machines,
it is not difficult to set a few config options in your user config file to
describe your machine.  Then, ``compass`` can use this data to make sure test
cases are configured in a way that is appropriate for your machine.

.. _dev_supported_machines:

Supported Machines
------------------

If you follow the procedure in :ref:`dev_conda_env`, you will have an
activation script for activating the development conda environment, setting
loading system modules and setting environment variables so you can build
MPAS and work with ``compass``.  Just source the script that should appear in
the base of your compass branch, e.g.:

.. code-block:: bash

    source load_dev_compass_1.0.0_anvil_intel_impi.sh

After loading this environment, you can set up test cases or test suites, and
a link ``load_compass_env.sh`` will be included in each suite or test case
work directory.  This is a link to the activation script that you sourced when
you were setting things up.  You can can source this file on a compute node
(e.g. in a job script) to get the right compass conda environment, compilers,
MPI libraries and environment variables for running ``compass`` tests and
the MPAS model.

.. note::

  Albany (and therefore most of the functionality in MALI) is currently only
  supported for those configurations with ``gnu`` compilers.


+--------------+------------+-----------+-------------------+
| Machine      | Compiler   | MPI lib.  |  MPAS make target |
+==============+============+===========+===================+
| anvil        | intel      | impi      | intel-mpi         |
|              |            +-----------+-------------------+
|              |            | openmpi   | ifort             |
|              +------------+-----------+-------------------+
|              | gnu        | openmpi   | gfortran          |
|              |            +-----------+-------------------+
|              |            | mvapich   | gfortran          |
+--------------+------------+-----------+-------------------+
| chicoma-cpu  | gnu        | mpich     | gnu-cray          |
+--------------+------------+-----------+-------------------+
| chrysalis    | intel      | openmpi   | ifort             |
|              |            +-----------+-------------------+
|              |            | impi      | intel-mpi         |
|              +------------+-----------+-------------------+
|              | gnu        | openmpi   | gfortran          |
+--------------+------------+-----------+-------------------+
| compy        | intel      | impi      | intel-mpi         |
|              +------------+-----------+-------------------+
|              | gnu        | openmpi   | gfortran          |
+--------------+------------+-----------+-------------------+
| cori-haswell | intel      | mpt       | intel-cray        |
|              +------------+-----------+-------------------+
|              | gnu        | mpt       | gnu-cray          |
+--------------+------------+-----------+-------------------+
| pm-cpu       | gnu        | mpich     | gnu-cray          |
+--------------+------------+-----------+-------------------+

Below are specifics for each supported machine

.. toctree::
   :titlesonly:

   anvil
   chicoma
   chrysalis
   compy
   cori
   perlmutter


.. _dev_other_machines:

Other Machines
--------------

If you are working on an "unknown" machine, the procedure is pretty similar
to what was described in :ref:`dev_conda_env`.  The main difference is that
we will use ``mpich`` or ``openmpi`` and the gnu compilers from conda-forge
rather than system compilers.  To create a development conda environment and
an activation script for it, on Linux, run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c gnu -i mpich

and on OSX run:

.. code-block:: bash

  ./conda/configure_compass_env.py --conda <conda_path> -c clang -i mpich

You may use ``openmpi`` instead of ``mpich`` but we have had better experiences
with the latter.

The result should be an activation script ``load_dev_compass_1.0.0_<mpi>.sh``.
Source this script to get the appropriate conda environment and environment
variables.

Under Linux, you can build the MPAS model with

.. code-block:: bash

    make gfortran

Under OSX, you can build the MPAS model with

.. code-block:: bash

    make gfortran-clang
