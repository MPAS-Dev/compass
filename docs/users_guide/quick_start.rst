.. _quick_start:

Quick Start for Users
=====================

.. _conda_env:

compass conda environment
-------------------------

.. note::

    **The following are planned instructions after a compass release.**

For each ``compass`` release, we maintain a
`conda environment <https://docs.conda.io/en/latest/>`_. that includes the
``compass`` package as well as all of its dependencies and some libraries
(currently `ESMF <https://earthsystemmodeling.org/>`_ and
`SCORPIO <https://e3sm.org/scorpio-parallel-io-library/>`_) built with system
MPI on our standard machines (Grizzly, Badger, Anvil, Chrysalis, Compy and
Cori).  Here are the commands to load the the environment for the latest
``compass`` release with the default compiler and MPI library on each machine:

* Anvil (Blues):

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_latest_compass.sh

* Grizzly and Badger:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_latest_compass.sh

* Chrysalis:

.. code-block:: bash

    source /lcrc/soft/climate/compass/chrysalis/load_latest_compass.sh

* Compy:

.. code-block:: bash

    source /share/apps/E3SM/conda_envs/compass/load_latest_compass.sh

* Cori-Haswell:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/compass/cori-haswell/load_latest_compass.sh

* Cori-KNL:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/compass/cori-knl/load_latest_compass.sh

Each of these paths has load scripts for the latest version of compass with
all supported compiler and MPI combinations.  For example, on Anvil, you can
get an environment appropriate for build MPAS components with Gnu compilers
and OpenMPI using:

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_latest_compass_gnu_openmpi.sh

To install your own ``compass`` conda environment on other machines, first,
install `Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ (if
miniconda is not already installed), then add the
`conda-forge channel <https://conda-forge.org/#about>`_:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict

then, create a new conda environment (called ``compass`` in this example) as
follows:

.. code-block:: bash

    conda create -n compass -c conda-forge -c e3sm/label/compass python=3.9 "compass=*=mpi_mpich*"

This will install the version of the package with MPI from conda-forge's MPICH
package.  If you want OpenMPI, use ``"compass=*=mpi_openmpi*"`` instead.  If
you do not want MPI from conda-forge (e.g. because you are working with a
system with its own MPI), use ``"compass=*=nompi*"``

To get a specific version of ``compass``, you can instead run:

.. code-block:: bash

    conda create -n compass -c conda-forge -c e3sm/label/compass python=3.9 "compass=1.0.0=mpi_mpich*"

That is, you will replace ``compass=*`` with ``compass=1.0.0``.  Each time you
want to work with compass, you will need to run:

.. code-block:: bash

    conda activate compass

Building MPAS components
------------------------

For instructions on how to build MPAS components, see the :ref:`dev_build_mpas`
section of the Developer's Guide.

.. _setup_overview:

Setting up test cases
---------------------

Before you set up a test case with ``compass``, you will need to build the
MPAS component you wish to test with.  Since the instructions for building
MPAS are machine specific, they are covered in the :ref:`machines` part of the
User's Guide.

To see all available test cases you can set up in compass, run:

.. code-block:: bash

    compass list

and you get output like this:

.. code-block:: none

   0: landice/circular_shelf/decomposition_test
   1: landice/dome/2000m/sia_smoke_test
   2: landice/dome/2000m/sia_decomposition_test
   3: landice/dome/2000m/sia_restart_test
   4: landice/dome/2000m/fo_smoke_test
   5: landice/dome/2000m/fo_decomposition_test
   6: landice/dome/2000m/fo_restart_test
   7: landice/dome/variable_resolution/sia_smoke_test
   8: landice/dome/variable_resolution/sia_decomposition_test
   9: landice/dome/variable_resolution/sia_restart_test
   ...

The list is long, so it will likely be useful to ``grep`` for particular
content:

.. code-block:: bash

    compass list | grep baroclinic_channel

.. code-block:: none

  32: ocean/baroclinic_channel/1km/rpe_test
  33: ocean/baroclinic_channel/4km/rpe_test
  34: ocean/baroclinic_channel/10km/rpe_test
  35: ocean/baroclinic_channel/10km/decomp_test
  36: ocean/baroclinic_channel/10km/default
  37: ocean/baroclinic_channel/10km/restart_test
  38: ocean/baroclinic_channel/10km/threads_test

See :ref:`dev_compass_list` for more information.

To set up a particular test case, you can either use the full path of the
test case:

.. code-block:: bash

    compass setup -t ocean/global_ocean/QU240/mesh -w <workdir> -p <mpas_path>

or you can replace the ``-t`` flag with the simple shortcut: ``-n 15``.  You
can set up several test cases at once by passing test numbers separated by
spaces: ``-n 15 16 17``.  See :ref:`dev_compass_setup` for more details.

Here, ``<workdir>`` is a path, usually to your scratch space. For example, on
Badger on LANL IC, you might use:

.. code-block:: bash

    -w /lustre/scratch4/turquoise/$USER/runs/210131_test_new_branch

The placeholder ``<mpas>`` is the relative or absolute path where the MPAS
component has been built (the directory, not the executable itself; see
:ref:`machines`).  You will typically want to provide a path either with ``-p``
or in a config file (see below) because the default paths are only useful for
developers running out of the ``compass`` repository.

You can explicitly specify a supported machine with ``-m <machine>``. You can
run:

.. code-block:: bash

    compass list --machines

to see what machines are currently supported. If you omit the ``-m`` flag,
``compass`` will try to automatically detect if you are running on a supported
machine and will fall back to a default configuration if no supported machine
is detected.

You may point to a config file with ``-f``:

.. code-block:: bash

    compass setup -t ocean/global_ocean/QU240/mesh -f ocean.cfg -w <workdir>

to specify config options that override the defaults from ``compass`` as a
whole, individual testcases, or machines.  If you are working on a supported
machine and you used ``-p`` to point to the MPAS build you want to use, you do
not need a config file.

If you are not on one of the supported machines, you will need to create a
config file like in this example for MPAS-Ocean. See also
`these examples <https://github.com/MPAS-Dev/compass/tree/master/example_configs>`_
in the repository.

.. code-block:: cfg

    # This file contains some common config options you might want to set
    # if you're working with the compass MPAS-Ocean or MALI.

    # The paths section describes paths that are used within landice and ocean
    # test cases.
    [paths]

    # The root to a location where data files for MALI will be cached
    landice_database_root = </path/to/landice_datafiles>

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = </path/to/ocean_databases>

    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = single_node

    # whether to use mpirun or srun to run the model
    parallel_executable = mpirun -host localhost

    # cores per node on the machine
    cores_per_node = 4

    # the number of multiprocessing or dask threads to use
    threads = 4

The two ``*_database_root`` directories can point to locations where you would
like to download data for MALI and MPAS-Ocean.  This data is downloaded only
once and cached for the next time you call ``compass setup`` or
``compass suite`` (see below).

The ``cores_per_node`` and ``threads`` config options should be the number of
CPUs on your computer.  You can set this to a smaller number if you want
``compass``.

In order to run regression testing that compares the output of the current run
with that from a previous compass run, use ``-b <previous_workdir>`` to specify
a "baseline".

When you set up one or more test cases, they will also be included in a custom
test suite, which is called ``custom`` by default.  (You can give it another
name with the ``--suite_name`` flag.)  You can run all the test cases in
sequence with one command as described in :ref:`suite_overview` or run them
one at a time as follows.

Running a test case
-------------------

After compiling the code and setting up a test case, you can log into an
interactive node (see :ref:`supported_machines`), load the required conda
environment and modules, and then

.. code-block:: bash

    cd <workdir>/<test_subdir>
    compass run

The ``<workdir>`` is the same path provided to the ``-w`` flag above.  The
sequence of subdirectories (``<test_subdir>``) is the same as given when you
list the test cases.  If the test case was set up properly, the directory
should contain a file ``test_case.pickle`` that contains the information
``compass`` needs to run the test case.

.. _suite_overview:

Test Suites
-----------

``compass`` includes several suites of test cases for code regressions and
bit-for-bit testing, as well as simply to make it easier to run several test
cases in one call. They can be listed with:

.. code-block:: bash

    compass list --suites

The output is:

.. code-block:: none

    Suites:
      -c landice -t fo_integration
      -c landice -t full_integration
      -c landice -t sia_integration
      -c ocean -t cosine_bell_cached_init
      -c ocean -t ec30to60
      -c ocean -t ecwisc30to60
      -c ocean -t nightly
      -c ocean -t pr
      -c ocean -t qu240_for_e3sm
      -c ocean -t quwisc240
      -c ocean -t quwisc240_for_e3sm
      -c ocean -t sowisc12to60
      -c ocean -t wc14

You can set up a suite as follows:

.. code-block:: bash

    compass suite -s -c ocean -t nightly -w <workdir> -p <mpas_path>

where the details are similar to setting up a case. You can use the same
config file (e.g. ``-f ocean.cfg``) and you can specify a "baseline" with
``-b <previous_workdir>`` for regression testing of the output compared with a
previous run of the ``nightly`` suite. See :ref:`dev_compass_suite` for more
on this command.

To run the regression suite, log into an interactive node, load your modules,
and

.. code-block:: bash

    cd <workdir>
    compass run [nightly]

In this case, you can specify the name of the suite to run.  This is required
if there are multiple suites in the same ``<workdir>``.  You can optionally
specify a suite like ``compass run [suitename].pickle``, which is convenient
for tab completion on the command line.
