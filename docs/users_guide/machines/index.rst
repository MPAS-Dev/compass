.. _machines:

Machines
========

One of the major advantages of ``compass`` over :ref:`legacy_compass` is that it
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

    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /usr/projects/regionalclimate/COMMON_MPAS/mpas_standalonedata/mpas-albany-landice


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the slurm account
    account = e3sm

The ``paths`` section provides local paths to the root of the "databases"
(local caches) of data files for each MPAS core.  These are generally in a
shared location for the project to save space.

The ``parallel`` section defined properties of the machine, to do with parallel
runs. Currently, machine files are defined for high-performance computing (HPC)
machines with multiple nodes.  These machines all use :ref:`slurm` to submit
parallel jobs.  They also all use the ``srun`` command to run individual
tasks within a job.  The number of ``cores_per_node`` vary between machines,
as does the account that typical ``compass`` users will have access to on the
machine.

.. _slurm:

Slurm job queueing
------------------

Most HPC systems now use the
`slurm workload manager <https://slurm.schedmd.com/documentation.html>`_.
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

.. note::

    Compass 1.0.0 has not yet been released.  The documentation for supported
    machines is what we anticipate once the release has occurred.  Developers
    or users working off of the `master <https://github.com/MPAS-Dev/compass/tree/master>`_
    branch (as opposed to a ``compass`` release), you should look at the
    developer's guide on :ref:`dev_supported_machines`.

On each supported machine, users will be able to source a script to activate
the appropriate compass environment and compilers.  Most machines support 2
compilers, each with one "flavor" of MPI and the required NetCDF, pNetCDF and
SCORPIO libraries.  These scripts will first load the conda environment for
``compass``, then it will load modules and set environment variables that will
allow you to build and run the MPAS model.

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

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /home/xylar/data/mpas/mpas_standalonedata/mpas-albany-landice


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

The paths for the MPAS core "databases" can be any emtpy path to begin with.
If the path doesn't exist, ``compass`` will create it.

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


