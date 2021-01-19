Chrysalis
=========

config options
--------------

Here are the default config options added when you choose ``-m chrysalis`` when
setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/mesh_database
    initial_condition_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/initial_condition_database
    bathymetry_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/bathymetry_database

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
    cores_per_node = 64

    # the number of multiprocessing or dask threads to use
    threads = 18


intel on Chrysalis
------------------

This is a stub for now.  More coming soon...

First, you might want to build SCORPIO (see below), or use the one from
`xylar <http://github.com/xylar>`_ referenced here:

.. code-block:: bash

    source /lcrc/soft/climate/e3sm-unified/load_latest_compass.sh



SCORPIO on Chrysalis
--------------------

This is a stub for now.  More coming soon...

If you need to compile it yourself, you can do that as follows (contact
`xylar <http://github.com/xylar>`_ if you run into trouble):

.. code-block:: bash

    #!/bin/bash

