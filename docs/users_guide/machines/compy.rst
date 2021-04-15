.. _machine_compy:

CompyMcNodeFace
===============

So far, we have not got ``compass`` to successfully build and run Compy.  We
will update this page when we do.

config options
--------------

Here are the default config options added when you choose ``-m compy`` when
setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /compyfs/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /compyfs/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /share/apps/E3SM/conda_envs/base


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

intel on CompyMcNodeFace
------------------------

Coming soon...

pgi on CompyMcNodeFace
----------------------

Coming soon...
