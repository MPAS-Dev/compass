.. _quick_start:

Quick Start for Users
=====================

.. _conda_env:

compass conda environment
-------------------------

E3SM supported machines
~~~~~~~~~~~~~~~~~~~~~~~

For each ``compass`` release, we maintain a
`conda environment <https://docs.conda.io/en/latest/>`_. that includes the
``compass`` package as well as all of its dependencies and some libraries
(currently `ESMF <https://earthsystemmodeling.org/>`_ and
`SCORPIO <https://e3sm.org/scorpio-parallel-io-library/>`_) built with system
MPI on our standard machines (Anvil, Chicoma, Chrysalis, Compy, and Perlmutter).
Here are the commands to load the the environment for the latest
``compass`` release with the default compiler and MPI library on each machine:

* Anvil (Blues):

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_latest_compass.sh

* Chicoma-CPU (coming soon):

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/chicoma-cpu/load_latest_compass.sh

* Chrysalis:

.. code-block:: bash

    source /lcrc/soft/climate/compass/chrysalis/load_latest_compass.sh

* Compy:

.. code-block:: bash

    source /share/apps/E3SM/conda_envs/compass/load_latest_compass.sh

* Perlmutter-CPU (coming soon):

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/compass/pm-cpu/load_latest_compass.sh


These same paths (minus ``load_latest_compass.sh``) also have load scripts for
the latest version of compass with all the supported compiler and MPI
combinations.  For example, on Anvil, you can get an environment appropriate
for build MPAS components with Gnu compilers and OpenMPI using:

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_latest_compass_gnu_openmpi.sh

other machines
~~~~~~~~~~~~~~

To install your own ``compass`` conda environment on other machines, first,
install `Miniforge3 <https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3>`_
(if it is not already installed), then create a new conda environment (called
``compass`` in this example) as follows:

.. code-block:: bash

    conda create -n compass -c conda-forge -c e3sm/label/compass python=3.10 \
        "compass=*=mpi_mpich*"

This will install the version of the package with MPI from conda-forge's MPICH
package.  If you want OpenMPI, use ``"compass=*=mpi_openmpi*"`` instead.  If
you do not want MPI from conda-forge (e.g. because you are working with a
system with its own MPI), use ``"compass=*=nompi*"``

To get a specific version of ``compass``, you can instead run:

.. code-block:: bash

    conda create -n compass -c conda-forge -c e3sm/label/compass python=3.9 \
        "compass=1.0.0=mpi_mpich*"

That is, you will replace ``compass=*`` with ``compass=1.0.0``.

Then, you will need to create a load script to activate the conda environment
and set some environment variables. In a directory where you want to store the
script, run:

.. code-block:: bash

    conda activate compass
    create_compass_load_script

From then on, each time you want to set up test cases or suites with compass
or build MPAS components, you will need to source that load script, for
example:

.. code-block:: bash

    source load_compass_1.0.0_mpich.sh

When you set up tests, a link called ``load_compass_env.sh`` will be added to
each test case or suite work directory.  To run the tests, you may find it
more convenient to source that link instead of finding the path to the original
load script.

.. _build_mpas:

Building MPAS components
------------------------

You will need to check out a branch of E3SM to build an MPAS component.

Typically, for MPAS-Ocean, you will clone
`E3SM <https://github.com/E3SM-Project/E3SM>`_ and for MALI, you will clone
`MALI-Dev <https://github.com/MALI-Dev/E3SM>`_.

To build MPAS-Ocean, first source the appropriate load script (see
:ref:`conda_env`) then run:

.. code-block:: bash

    cd components/mpas-ocean
    git submodule update --init --recursive
    make <mpas_make_target>

MALI can be compiled with or without the Albany library that contains the
first-order velocity solver.  The Albany first-order velocity solver is the
only velocity option that is scientifically validated, but the Albany library
is only available with Gnu compilers (and therefore not at all on Compy).
Therefore, in some situations it is desirable to compile without Albany to run
basic tests on platforms where Albany is not available.  This basic mode of
MALI can be compiled similarly to MPAS-Ocean.  Again, first source the
appropriate load script (see :ref:`conda_env`) then run:

.. code-block:: bash

    cd components/mpas-albany-landice
    git submodule update --init --recursive
    make [ALBANY=true] <mpas_make_target>

where `ALBANY=true` is included if you want to compile with Albany support
and excluded if you do not.  Some more information on building and running
MALI is available at
`https://github.com/MALI-Dev/E3SM/wiki <https://github.com/MALI-Dev/E3SM/wiki>`_.

See the last column of the table in :ref:`dev_supported_machines` for the right
``<mpas_make_target>`` command for each machine and compiler.


.. _setup_overview:

Setting up test cases
---------------------

Before you set up a test case with ``compass``, you will need to build the
MPAS component you wish to test with, see :ref:`build_mpas` above.

If you have not already done so, you will need to source the appropriate load
script, see :ref:`conda_env`.

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
Chrysalis at LCRC, you might use:

.. code-block:: bash

    -w /lcrc/group/e3sm/$USER/runs/210131_test_new_branch

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
config file like in this example. See also
`this example <https://github.com/MPAS-Dev/compass/tree/main/example_configs>`_
in the repository.

.. code-block:: cfg

    # This file contains some common config options you might want to set

    # The paths section describes paths to databases and shared compass environments
    [paths]

    # A root directory where MPAS standalone data can be found
    database_root = </path/to/root>/mpas_standalonedata

    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = single_node

    # whether to use mpirun or srun to run the model
    parallel_executable = mpirun -host localhost

    # cores per node on the machine, detected automatically by default
    # cores_per_node = 4

The ``database_root`` directory can point to a location where you would like to
download data for MALI, MPAS-Seaice and MPAS-Ocean.  This data is downloaded
only once and cached for the next time you call ``compass setup`` or
``compass suite`` (see below).

The ``cores_per_node`` config option will default to the number of CPUs on your
computer.  You can set this to a smaller number if you want ``compass`` to
use fewer cores.

In order to run regression testing that compares the output of the current run
with that from a previous compass run, use ``-b <previous_workdir>`` to specify
a "baseline".

When you set up one or more test cases, they will also be included in a custom
test suite, which is called ``custom`` by default.  (You can give it another
name with the ``--suite_name`` flag.)  You can run all the test cases in
sequence with one command as described in :ref:`suite_overview` or run them
one at a time as follows.

If you want to copy the MPAS executable over to the work directory, you can
use the ``--copy_executable`` flag or set the config option
``copy_executable = True`` in the ``[setup]`` section of your user config
file.  One use of this capability for compass simulations that are used in
a paper.  In that case, it would be better to have a copy of the executable
that will not be changed even if the E3SM branch is modified, recompiled or
deleted.  Another use might be to maintain a long-lived baseline test.
Again, it is safer to have the executable used to produce the baseline
preserved.

Running a test case
-------------------

After compiling the code and setting up a test case, you can log into an
interactive node (see :ref:`supported_machines`), load the required conda
environment and modules, and then

.. code-block:: bash

    cd <workdir>/<test_subdir>
    source load_compass_env.sh
    compass run

The ``<workdir>`` is the same path provided to the ``-w`` flag above.  The
sequence of subdirectories (``<test_subdir>``) is the same as given when you
list the test cases.  If the test case was set up properly, the directory
should contain a file ``test_case.pickle`` that contains the information
``compass`` needs to run the test case.  The load script
``load_compass_env.sh`` is a link to whatever load script you sourced before
setting up the test case (see :ref:`conda_env`).

Running with a job script
-------------------------

Alternatively, on supported machines, you can run the test case or suite with
a job script generated automatically during setup, for example:

.. code-block:: bash

    cd <workdir>/<test_subdir>
    sbatch job_script.sh

You can edit the job script to change the wall-clock time (1 hour by default)
or the number of nodes (scaled according to the number of cores require by the
test cases by default).

.. code-block:: bash

    #!/bin/bash
    #SBATCH  --job-name=compass
    #SBATCH  --account=condo
    #SBATCH  --nodes=5
    #SBATCH  --output=compass.o%j
    #SBATCH  --exclusive
    #SBATCH  --time=1:00:00
    #SBATCH  --qos=regular
    #SBATCH  --partition=acme-small


    source load_compass_env.sh
    compass run

You can also use config options, passed to ``compass suite`` or
``compass setup`` with ``-f`` in a user config file to control the job script.
The following are the config options that are relevant to job scripts:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # account for running diagnostics jobs
    account = condo

    # Config options related to creating a job script
    [job]

    # the name of the parallel job
    job_name = compass

    # wall-clock time
    wall_time = 1:00:00

    # The job partition to use, by default, taken from the first partition (if any)
    # provided for the machine by mache
    partition = acme-small

    # The job quality of service (QOS) to use, by default, taken from the first
    # qos (if any) provided for the machine by mache
    qos = regular

    # The job constraint to use, by default, taken from the first constraint (if
    # any) provided for the  machine by mache
    constraint =


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
      -c landice -t calving_dt_convergence
      -c landice -t fo_integration
      -c landice -t full_integration
      -c landice -t humboldt_calving_tests
      -c landice -t humboldt_calving_tests_fo
      -c landice -t sia_integration
      -c ocean -t cosine_bell_cached_init
      -c ocean -t ec30to60
      -c ocean -t ecwisc30to60
      -c ocean -t kuroshio8to60
      -c ocean -t kuroshio12to60
      -c ocean -t nightly
      -c ocean -t pr
      -c ocean -t qu240_for_e3sm
      -c ocean -t quwisc240
      -c ocean -t quwisc240_for_e3sm
      -c ocean -t so12to30
      -c ocean -t sowisc12to30
      -c ocean -t wc14
      -c ocean -t wcwisc14
      -c ocean -t wetdry

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
    source load_compass_env.sh
    compass run [nightly]

In this case, you can specify the name of the suite to run.  This is required
if there are multiple suites in the same ``<workdir>``.  You can optionally
specify a suite like ``compass run [suitename].pickle``, which is convenient
for tab completion on the command line. The load script
``load_compass_env.sh`` is a link to whatever load script you sourced before
setting up the test case (see :ref:`conda_env`).