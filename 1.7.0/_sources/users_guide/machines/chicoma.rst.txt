.. _machine_chicoma:

Chicoma
=======

`LANL IC overview and search <https://int.lanl.gov/hpc/institutional-computing/index.shtml>`_

`DST Calendar <http://hpccalendar.lanl.gov/>`_ (within LANL network)

Information about Slurm:

* `Introduction to Slurm at LANL <https://hpc.lanl.gov/job-scheduling/index.html#JobScheduling-IntroductiontoSlurm>`_

* `Basic Slurm Guide for LANL HPC Users <https://hpc.lanl.gov/job-scheduling/basic-slurm-guide-for-lanl-hpc-users.html>`_

* `Slurm Command Summary <https://hpc.lanl.gov/job-scheduling/slurm-commands.html>`_

* `Slurm: Running Jobs on HPC Platforms <https://hpc.lanl.gov/job-scheduling/slurm-commands.html#SlurmCommands-SlurmJobSubmission>`_

* `example of batch scripts <https://hpc.lanl.gov/job-scheduling/basic-slurm-guide-for-lanl-hpc-users.html#BasicSlurmGuideforLANLHPCUsers-BatchScriptGenerator>`_

Machine specifications: `chicoma <https://hpc.lanl.gov/platforms/chicoma/index.html>`_
`turquoise network <https://hpc.lanl.gov/networks/turquoise-network/index.html>`_

login: ``ssh -t <username>@wtrw.lanl.gov ssh ch-fe``

File locations:

* small home directory, for start-up scripts only: ``/users/<username>``

* home directory, backed up: ``/usr/projects/climate/<username>``

* scratch space, not backed up: ``/lustre/scratch4/turquoise/<username>`` or
  ``scratch5``

Check compute time:

* ``sacctmgr list assoc user=<username> format=Cluster,Account%18,Partition,QOS%45``

* Which is my default account? ``sacctmgr list user <username>``

* ``sshare -a | head -2; sshare -a | grep $ACCOUNT | head -1``

* ``sreport -t Hours cluster AccountUtilizationByUser start=2019-12-02 | grep $ACCOUNT``

* check job priority: ``sshare -a | head -2; sshare -a | grep $ACCOUNT``

* `LANL Cluster Usage Overview <https://hpcinfo.lanl.gov>`_ (within LANL yellow)

Check disk usage:

* your home space: ``chkhome``

* total disk usage in Petabytes: ``df -BP |head -n 1; df -BP|grep climate; df -BP |grep scratch``

Archiving

* `turquoise HPSS archive <https://hpc.lanl.gov/data/filesystems-and-storage-on-hpc-clusters/hpss-data-archive/index.html>`_

* archive front end: ``ssh -t <username>@wtrw.lanl.gov ssh ar-tn``

* storage available at: ``cd /archive/<project_name>``

* you can just copy files directly into here for a particular project.

LANL uses slurm. To obtain an interactive node:

.. code-block:: bash

    salloc -N 1 -t 2:0:0 --qos=interactive

Use ``--account=ACCOUNT_NAME`` to change to a particular account.

Chicoma-CPU
-----------

**There has not yet been a release with Chicoma-CPU, so the following applies
to the release of compass v1.2.0, when it happens.**

Chicoma's CPU and GPU nodes have different configuration options and compilers.
We only support Chicoma-CPU at this time.

config options
~~~~~~~~~~~~~~

Here are the default config options added when you choose ``-m chicoma-cpu``
when setting up test cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # A shared root directory where MPAS standalone data can be found
    database_root = /usr/projects/regionalclimate/COMMON_MPAS/mpas_standalonedata/

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/compass/chicoma-cpu/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = gnu

    # the system MPI library to use for gnu compiler
    mpi_gnu = mpich

    # the base path for spack environments used by compass
    spack = /usr/projects/climate/SHARED_CLIMATE/compass/chicoma-cpu/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    use_e3sm_hdf5_netcdf = True


    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # account for running diagnostics jobs
    account =

    # cores per node on the machine
    cores_per_node = 128

    # threads per core (set to 1 because trying to hyperthread seems to be causing
    # hanging on chicoma)
    threads_per_core = 1


    # Config options related to creating a job script
    [job]

    # The job partition to use
    partition = standard

    # The job reservation to use (needed for debug jobs)
    reservation =

    # The job quality of service (QOS) to use
    qos = standard
    

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
    cores_per_node = 256

    # available partition(s) (default is the first)
    partitions = standard, gpu

    # quality of service (default is the first)
    qos = standard, debug


    # Config options related to spack environments
    [spack]

    # whether to load modules from the spack yaml file before loading the spack
    # environment
    modules_before = False

    # whether to load modules from the spack yaml file after loading the spack
    # environment
    modules_after = False

Hyperthreading
~~~~~~~~~~~~~~

By default, hyperthreading has been disable on Chicoma. We had found some
some issues with runs hanging in early testing that seemed to be mitigated by
disabling hyperthreading.  We disable hyperthreading by setting
``threads_per_core = 1`` and reducing ``cores_per_node`` to not include the
2 hyperthreads.  You can re-enable hyperthreading on Chicoma by providing a
user config file where you set ``threads_per_core`` and ``cores_per_node``
as follows:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # cores per node on the machine (including hyperthreading)
    cores_per_node = 256

    # threads per core with hyperthreading
    threads_per_core = 2

Gnu on Chicoma-CPU
~~~~~~~~~~~~~~~~~~

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /usr/projects/climate/SHARED_CLIMATE/compass/chicoma-cpu/load_latest_compass_gnu_mpich.sh

To build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gnu-cray
