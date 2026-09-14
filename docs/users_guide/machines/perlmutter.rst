.. _machine_perlmutter:

Perlmutter
==========

login: ``ssh $my_username@perlmutter-p1.nersc.gov``

interactive login:

.. code-block:: bash

    # for CPU:
    salloc --partition=debug --nodes=1 --time=30:00 -C cpu

    # for GPU:
    salloc --partition=debug --nodes=1 --time=30:00 -C gpu

Compute time:

* Check hours of compute usage at https://iris.nersc.gov/

File system:

* Overview: https://docs.nersc.gov/filesystems/

* home directory: ``$HOME``

* scratch directory: ``$SCRATCH``

* Check your individual disk usage with ``myquota``

* Check the group disk usage with ``prjquota  projectID``, i.e.
  ``prjquota  m1795`` or ``prjquota  e3sm``

Archive:

* NERSC uses HPSS with the commands ``hsi`` and ``htar``

* overview: https://docs.nersc.gov/filesystems/archive/

* E3SM uses `zstash <https://e3sm-project.github.io/zstash/>`_


Perlmutter-CPU
--------------

Perlmutter's CPU and GPU nodes have different configuration options and
compilers.  Compass supports both: ``pm-cpu`` for MPAS-Ocean and MALI, and
``pm-gpu`` for MALI with the GPU-enabled Albany library.  The config options
for ``pm-gpu`` are in ``compass/machines/pm-gpu.cfg``.

config options
~~~~~~~~~~~~~~

Here are the default
config options added when you choose ``-m pm-cpu`` when setting up test
cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # A shared root directory where MPAS standalone data can be found
    database_root = /global/cfs/cdirs/e3sm/mpas_standalonedata

    # the path where shared compass environments are deployed
    compass_envs = /global/common/software/e3sm/compass/pm-cpu/base


    # Options related to deploying compass environments on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = gnu

    # the compiler to use to build software (e.g. ESMF and MOAB) with spack
    software_compiler = gnu

    # the system MPI library to use for gnu compiler
    mpi_gnu = mpich

    # the base path for spack environments used by compass
    spack = /global/cfs/cdirs/e3sm/software/compass/pm-cpu/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    use_e3sm_hdf5_netcdf = True

    # The parallel section describes options related to running jobs in parallel.
    # Most options in this section come from mache so here we just add or override
    # some defaults
    [parallel]

    # cores per node on the machine
    cores_per_node = 128

    # threads per core (set to 1 because trying to hyperthread seems to be causing
    # hanging on perlmutter)
    threads_per_core = 1

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

    # account for running diagnostics jobs
    account = e3sm

    # available constraint(s) (default is the first)
    constraints = cpu

    # quality of service (default is the first)
    qos = regular, premium, debug

    # Config options related to spack environments
    [spack]

    # whether the machine uses cray compilers
    cray_compilers = True

Hyperthreading
~~~~~~~~~~~~~~

By default, hyperthreading has been disable on Perlmutter. We had found some
some issues with runs hanging in early testing that seemed to be mitigated by
disabling hyperthreading.  We disable hyperthreading by setting
``threads_per_core = 1`` and reducing ``cores_per_node`` to not include the
2 hyperthreads.  You can re-enable hyperthreading on Perlmutter by providing a
user config file where you set ``threads_per_core`` and ``cores_per_node``
as follows:

.. code-block:: cfg

    # The parallel section describes options related to running jobs in parallel
    [parallel]

    # cores per node on the machine (including hyperthreading)
    cores_per_node = 256

    # threads per core with hyperthreading
    threads_per_core = 2

Loading and running compass on Perlmutter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the Developer's Guide at :ref:`dev_machine_perlmutter` to deploy
``compass`` and build MPAS components.  There are currently no shared
``compass`` environments for users on Perlmutter.


Jupyter notebook on remote data
-------------------------------

You can run Jupyter notebooks on NERSC with direct access to scratch data as
follows:

.. code-block:: bash

    ssh -Y -L 8844:localhost:8844 MONIKER@perlmutter-p1.nersc.gov
    jupyter notebook --no-browser --port 8844
    # in local browser, go to:
    http://localhost:8844/

Note that on NERSC, you can also use their
`Jupyter server <https://jupyter.nersc.gov/>`_,
it’s really nice and grabs a compute node for you automatically on logon.
You’ll need to create a python kernel from e3sm-unified following these steps
(taken from https://docs.nersc.gov/connect/jupyter/).  After creating the
kernel, you just go to “Change Kernel” in the Jupyter notebook and you’re ready
to go.

You can use one of NERSC's default Python 3 or R kernels. If you have a
Conda environment, depending on how it is installed, it may just show up in the
list of kernels you can use. If not, use the following procedure to enable a
custom kernel based on a Conda environment. Let's start by assuming you are a
user with username ``user`` who wants to create a Conda environment on
Perlmutter and use it from Jupyter.

.. code-block:: bash


    module load python
    conda create -n myenv python=3.7 ipykernel <further-packages-to-install>
    <... installation messages ...>
    source activate myenv
    python -m ipykernel install --user --name myenv --display-name MyEnv
       Installed kernelspec myenv in /global/u1/u/user/.local/share/jupyter/kernels/myenv

Be sure to specify what version of Python interpreter you want installed. This
will create and install a JSON file called a "kernel spec" in ``kernel.json`` at
the path described in the install command output.

.. code-block:: json

    {
        "argv": [
            "/global/homes/u/user/.conda/envs/myenv/bin/python",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        ],
        "display_name": "MyEnv",
        "language": "python"
    }
