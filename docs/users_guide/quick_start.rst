.. _quick_start:

Quick Start for Users
=====================

.. _conda_env:

compass conda environment
-------------------------

The compass conda environment includes all the python libraries needed to run
compass scripts. It is maintained on our standard machines (Grizzly, Badger,
Anvil, Compy and Cori).  Here are the commands for each machine:

* grizzly and badger:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/load_latest_compass.sh

* anvil (blues) and chrysalis:

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh

* compy:

.. code-block:: bash

    source /share/apps/E3SM/conda_envs/load_latest_compass.sh

* cori:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/anaconda_envs/load_latest_compass.sh

To install your own compass conda environment on other machines, first, install
`Miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ (if miniconda is
not already installed), then add the
`conda-forge channel <https://conda-forge.org/#about>`_:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict

then create a new conda environment (called ``compass`` in this example) as
follows:

.. code-block:: bash

    conda create -n compass python=3.8 compass=1.0.0

Each time you want to work with compass, you will need to run:

.. code-block:: bash

    conda activate compass

.. _setup_overview:

Setting up a test case
----------------------

To see all available test cases you can set up in compass, run:

.. code-block:: bash

    compass list

and you get output like this:

.. code-block:: none

  15: ocean/global_ocean/QU240/mesh
  16: ocean/global_ocean/QU240/PHC/init
  17: ocean/global_ocean/QU240/PHC/performance_test/split_explicit
  18: ocean/global_ocean/QU240/PHC/performance_test/RK4

The list is long, so it will likely be useful to grep for particular content:

.. code-block:: bash

    compass list | grep baroclinic_channel

.. code-block:: none

   8: ocean/baroclinic_channel/1km/rpe_test
   9: ocean/baroclinic_channel/4km/rpe_test
  10: ocean/baroclinic_channel/10km/rpe_test
  11: ocean/baroclinic_channel/10km/decomp_test
  12: ocean/baroclinic_channel/10km/default
  13: ocean/baroclinic_channel/10km/restart_test
  14: ocean/baroclinic_channel/10km/threads_test

See :ref:`dev_compass_list` for more information.

To set up a particular test case, you can either use the full path of the
test case:

.. code-block:: bash

    compass setup -t ocean/global_ocean/QU240/mesh -m $MACHINE -w $WORKDIR

or you can replace the ``-t`` flag with the simple shortcut: ``-n 15``.  You
can set up several test cases at once by passing test numbers separated by
spaces: ``-n 15 16 17``.  See :ref:`dev_compass_setup` for more details.

Here ``$WORKDIR`` is a path, usually to your scratch space. For example,

.. code-block:: bash

    -w /lustre/scratch4/turquoise/$USER/runs/210131_test_new_branch

``$MACHINE`` is one of the known machines (omit the ``-m`` flag if you are not
working on one of the known machines).  You can run:

.. code-block:: bash

    compass list --machines

to see what machines are currently supported.

You may name a config file with ``-f``:

.. code-block:: bash

    compass setup -t ocean/global_ocean/QU240/mesh -f ocean.cfg -w $WORKDIR

to specify config options that override the defaults from compass as a whole,
individual testcases, or machines.  If you are working on a supported machine
and running MPAS-Model out of the default directory for your MPAS component
(e.g. ``MPAS-Model/ocean/develop``), you do not need a config file.

If you are not on one of the supported machines, you will need to create a
config file like in this example for MPAS-Ocean. See also
`these examples <https://github.com/MPAS-Dev/compass/tree/master/example_configs>`_
in the repository.

.. code-block:: cfg

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS-Ocean
    # has been built
    mpas_model = MPAS-Model/ocean/develop

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /home/xylar/data/mpas/meshes
    initial_condition_database = /home/xylar/data/mpas/initial_conditions
    bathymetry_database = /home/xylar/data/mpas/bathymetry_database


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = single_node

    # whether to use mpirun or srun to run the model
    parallel_executable = mpirun

    # cores per node on the machine
    cores_per_node = 8

    # the number of multiprocessing or dask threads to use
    threads = 8

Running a test case
-------------------

After compiling the code and setting up a test case, you can log into an
interactive node (see :ref:`supported_machines`) and then

.. code-block:: bash

    cd $WORKDIR
    ./run.py

Note the sequence of subdirectories is the same as given when you list the
test cases.

In order to run a bit-for-bit test with a previous case, use
``-b $PREVIOUS_WORKDIR`` to specify a "baseline".


.. _suite_overview:

Test Suites
-----------

compass includes several suites of test cases for code regressions and
bit-for-bit testing, as well as simply to make it easier to run several test
cases in one call. For the ocean core, they can be listed with:

.. code-block:: bash

    compass list --suites

You can set up a suite as follows:

.. code-block:: bash

    compass suite -s -c ocean -t nightly -m $MACHINE -w $WORKDIR

where the details are similar to setting up a case. You can use the same
config file (e.g. ``-f ocean.cfg``) and you can specify a "baseline" with
``-b $PREVIOUS_WORKDIR`` for bit-for-bit comparison of the output with a
previous run of the ``nightly`` suite. See :ref:`dev_compass_suite` for more
on this command.

To run the regression suite, log into an interactive node, load your modules,
and

.. code-block:: bash

    cd $WORKDIR
    ./nightly.py
