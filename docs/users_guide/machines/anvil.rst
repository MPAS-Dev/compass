.. _machine_anvil:

Anvil
=====

Anvil is a set of nodes used by E3SM and its "ecosystem" projects on
``blues`` at `LCRC <https://www.lcrc.anl.gov/>`_.  To gain access to the
machine, you will need access to E3SM's confluence pages or the equivalent for
your ecosystem project.

config options
--------------

Here are the default config options added when you choose ``-m anvil`` when
setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /lcrc/soft/climate/e3sm-unified/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the number of multiprocessing or dask threads to use
    threads = 18

intel18 on anvil
----------------

.. note::

    Compass 1.0.0 has not yet been released.  The following will apply after
    the release.

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_compass1.0.0_intel18_mvapich.sh

To build the MPAS model with

.. code-block:: bash

    make CORE=landice ifort

or

.. code-block:: bash

    make CORE=ocean ifort

gnu on anvil
------------

.. note::

    Compass 1.0.0 has not yet been released.  The following will apply after
    the release.

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /lcrc/soft/climate/compass/anvil/load_compass1.0.0_gnu_mvapich.sh

To build the MPAS model with

.. code-block:: bash

    make CORE=landice gfortran

or

.. code-block:: bash

    make CORE=ocean gfortran
