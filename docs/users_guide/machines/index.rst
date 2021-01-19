.. _machines:

Machines
========

One of the major advantages of compass over :ref:`legacy_compass` is that it
attempts to be aware of the capabilities of the machine it is running on.  This
is a particular advantage for so-called "supported" machines with a config file
defined for them in the ``compass`` package.  But even for "unknown" machines,
it is not difficult to set a few config options in your user config file to
describe your machine.  Then, ``compass`` can use this data to make sure test
cases are configured in a way that is appropriate for your machine.

config options
--------------

The config options typically defined for a machine are:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/mesh_database
    initial_condition_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/initial_condition_database
    bathymetry_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/bathymetry_database

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the slurm account
    account = climateacme

    # the number of multiprocessing or dask threads to use
    threads = 18

The ``paths`` section provides local paths for 3 "databases" (local caches) of
data files used in compass test cases.  These are generally in a shared
location for the project to save space.

the ``compass_envs`` section will be used in the future to include activation
of the appropriate
`conda environment <https://docs.conda.io/projects/conda/en/latest/index.html>`_
for compass in automatically generated job scripts.

The ``parallel`` section defined properties of the machine, to do with parallel
runs. Currently, machine files are defined for high-performance computing (HPC)
machines with multiple nodes.  These machines all use :ref:`slurm` to submit
parallel jobs.  They also all use the ``srun`` command to run individual
tasks within a job.  The number of ``cores_per_node`` vary between machines,
as does the account that typical compass users will have access to on the
machine.

.. note::

    The number of ``threads`` to use in python multi-threaded operations is a
    placeholder option for future work on creating masks and performing other
    tasks in parallel in python instead of in serial with Fortran codes.

.. _slurm:

Slurm job queueing
------------------

Most HPC systems now use `slurm <https://slurm.schedmd.com/documentation.html>`_.
Here are some basic commands:

.. code-block:: bash

    salloc -N 1 -t 2:0:0 # interactive job (see machine specific versions below)
    sbatch script # submit a script
    squeue # show all jobs
    squeue -u $my_moniker # show only your jobs
    scancel jobID # cancel a job

.. _supported_machines:

Supported Machines
------------------

.. toctree::
   :titlesonly:

   anvil
   badger
   chrysalis
   compy
   cori
   grizzly


.. _other_machines:

Other Machines
--------------

If you are working on an "unknown" machine, you will need to define some of
the config options that would normally be in a machine's config file yourself
in your user config file:

.. code-block:: cfg

    [paths]

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

The paths for the 3 "databases" can be any emtpy path to being with.  If the
path doesn't exist, compass will create it.

If you're not working on an HPC machine, you will probably not have multiple
nodes or :ref:`slurm`.  You will probably install
`OpenMPI <https://www.open-mpi.org/>`_ or `MPICH <https://www.mpich.org/>`_,
possibly via a
`conda environment <https://docs.conda.io/projects/conda/en/latest/index.html>`_.
In this case, the ``parallel_executable`` is ``mpirun``.

.. toctree::
   :titlesonly:

   linux
   osx


