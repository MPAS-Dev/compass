Cori
====

login: ``ssh $my_username@cori.nersc.gov``

interactive login:

.. code-block:: bash

    # for Haswell:
    salloc --partition=debug --nodes=1 --time=30:00 -C haswell

    # for KNL:
    salloc --partition=debug --nodes=1 --time=30:00 -C knl

Compute time:

* Check hours of compute usage at https://nim.nersc.gov/

File system:

* Overview: https://docs.nersc.gov/filesystems/

* home directory: ``/global/homes/$my_username``

* scratch directory: ``/global/cscratch1/sd/$my_username``

* Check your individual disk usage with ``myquota``

* Check the group disk usage with ``prjquota  projectID``, i.e.
  ``prjquota  m2833`` or ``prjquota  acme``

Archive:

* NERSC uses HPSS with the commands ``hsi`` and ``htar``

* overview: https://docs.nersc.gov/filesystems/archive/

* E3SM uses `zstash <https://e3sm-project.github.io/zstash/docs/html/index.html>`_


Cori-Haswell
------------

Cori's Haswell and KNL nodes have different configuration options and
compilers.  We only support Cori-Haswell at this time.

config options
~~~~~~~~~~~~~~

Here are the default
config options added when you choose ``-m cori-haswell`` when setting up test
cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # A shared root directory where MPAS standalone data can be found
    database_root = /global/cfs/cdirs/e3sm/mpas_standalonedata

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /global/common/software/e3sm/compass/cori-haswell/base


    # Options related to deploying a compass conda environment on supported
    # machines
    [deploy]

    # the compiler set to use for system libraries and MPAS builds
    compiler = intel

    # the system MPI library to use for intel compiler
    mpi_intel = mpt

    # the system MPI library to use for gnu compiler
    mpi_gnu = mpt

    # the base path for spack environments used by compass
    spack = /global/cfs/cdirs/e3sm/software/compass/cori-haswell/spack

    # whether to use the same modules for hdf5, netcdf-c, netcdf-fortran and
    # pnetcdf as E3SM (spack modules are used otherwise)
    use_e3sm_hdf5_netcdf = True

    # the version of ESMF to build if using system compilers and MPI (don't build)
    esmf = None


    # Config options related to creating a job script
    [job]

    # The job constraint to use, by default, taken from the first constraint (if
    # any) provided for the  machine by mache
    constraint = haswell

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
    cores_per_node = 32

    # account for running diagnostics jobs
    account = e3sm

    # available configurations(s) (default is the first)
    configurations = haswell

    # quality of service (default is the first)
    qos = regular, premium, debug


Intel on Cori-Haswell
~~~~~~~~~~~~~~~~~~~~~

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/compass/cori-haswell/load_latest_compass_intel_mpt.sh

To build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] intel-cray


Gnu on Cori-Haswell
~~~~~~~~~~~~~~~~~~~

To load the compass environment and modules, and set appropriate environment
variables:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/compass/cori-haswell/load_latest_compass_gnu_mpt.sh

To build the MPAS model with

.. code-block:: bash

    make [DEBUG=true] [OPENMP=true] [ALBANY=true] gnu-cray


Jupyter notebook on remote data
-------------------------------

You can run Jupyter notebooks on NERSC with direct access to scratch data as
follows:

.. code-block:: bash

    ssh -Y -L 8844:localhost:8844 MONIKER@cori.nersc.gov
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

You can use one of our default Python 2, Python 3, or R kernels. If you have a
Conda environment, depending on how it is installed, it may just show up in the
list of kernels you can use. If not, use the following procedure to enable a
custom kernel based on a Conda environment. Let's start by assuming you are a
user with username ``user`` who wants to create a Conda environment on Cori and use
it from Jupyter.

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
