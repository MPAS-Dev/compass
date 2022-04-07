.. _machine_badger:

Badger
======

`LANL IC overview and search <https://int.lanl.gov/hpc/institutional-computing/index.shtml>`_

`DST Calendar <http://hpccalendar.lanl.gov/>`_ (within LANL yellow)

Information about Slurm:

* `Introduction to Slurm at LANL <https://hpc.lanl.gov/job-scheduling/index.html#JobScheduling-IntroductiontoSlurm>`_

* `Basic Slurm Guide for LANL HPC Users <https://hpc.lanl.gov/job-scheduling/basic-slurm-guide-for-lanl-hpc-users.html>`_

* `Slurm Command Summary <https://hpc.lanl.gov/job-scheduling/slurm-commands.html>`_

* `Slurm: Running Jobs on HPC Platforms <https://hpc.lanl.gov/job-scheduling/slurm-commands.html#SlurmCommands-SlurmJobSubmission>`_

* `example of batch scripts <https://hpc.lanl.gov/job-scheduling/basic-slurm-guide-for-lanl-hpc-users.html#BasicSlurmGuideforLANLHPCUsers-BatchScriptGenerator>`_

Machine specifications: `badger <https://hpc.lanl.gov/platforms/badger.html>`_
`turquoise network <https://hpc.lanl.gov/networks/turquoise-network/index.html>`_

login: ``ssh -t $my_moniker@wtrw.lanl.gov ssh ba-fe``

File locations:

* small home directory, for start-up scripts only: ``/users/$my_moniker``

* home directory, backed up: ``/usr/projects/climate/$my_moniker``

* scratch space, not backed up: ``/lustre/scratch3/turquoise/$my_moniker`` or
  ``scratch4``

Check compute time:

* ``sacctmgr list assoc user=$my_moniker format=Cluster,Account%18,Partition,QOS%45``

* Which is my default account? sacctmgr list user $my_moniker

* ``sshare -a | head -2; sshare -a | grep $ACCOUNT | head -1``

* ``sreport -t Hours cluster AccountUtilizationByUser start=2019-12-02 | grep $ACCOUNT``

* check job priority: ``sshare -a | head -2; sshare -a | grep $ACCOUNT``

* `LANL Cluster Usage Overview <https://hpcinfo.lanl.gov>`_ (within LANL yellow)

Check disk usage:

* your home space: ``chkhome``

* total disk usage in Petabytes: ``df -BP |head -n 1; df -BP|grep climate; df -BP |grep scratch``

Archiving

* `turquoise HPSS archive <https://hpc.lanl.gov/data/filesystems-and-storage-on-hpc-clusters/hpss-data-archive/index.html>`_

* archive front end: ``ssh -t $my_moniker@wtrw.lanl.gov ssh ar-tn``

* storage available at: ``cd /archive/<project_name>``

* you can just copy files directly into here for a particular project.

LANL uses slurm. To obtain an interactive node:

.. code-block:: bash

    salloc -N 1 -t 2:0:0 --qos=interactive

Use ``--account=ACCOUNT_NAME`` to change to a particular account.


config options
--------------

Here are the default config options added when Badger is automatically
detected or you choose ``-m badger`` when setting up test cases or a test
suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /usr/projects/regionalclimate/COMMON_MPAS/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/compass/badger/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = impi

    # the system MPI library to use for gnu compiler
    mpi_gnu = mvapich

    # the base path to system libraries to be added as part of setting up compass
    system_libs = /usr/projects/climate/SHARED_CLIMATE/compass/badger/system

Additionally, some relevant config options come from the
`mache <https://github.com/E3SM-Project/mache/>`_ package:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # parallel system of execution: slurm, cobalt or single_node
    system = slurm

    # whether to use mpirun or srun to run a task
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # account for running diagnostics jobs
    account = e3sm

    # quality of service (default is the first)
    qos = regular, interactive

Intel on Badger
---------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_latest_compass_intel_impi.sh


To build the MPAS model with

.. code-block:: bash

    make intel-mpi

Gnu on Badger
-------------

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/badger/load_latest_compass_gnu_mvapich.sh


To build the MPAS model with

.. code-block:: bash

    make gfortran
