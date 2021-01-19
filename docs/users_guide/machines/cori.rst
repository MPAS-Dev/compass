Cori
====

login: ``ssh $my_username@cori.nersc.gov``

compass environment:

.. code-block:: bash

    source /global/cfs/cdirs/e3sm/software/anaconda_envs/load_latest_compass.sh

example compass config file:
`general.config.ocean_cori <https://gist.github.com/mark-petersen/c61095d65216415ee0bb62a76da3c6cb>`_

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


config options
--------------

Since Cori's Haswell and KNL nodes have different configuration options, they
are treated as separate supported machines in compass.  Here are the default
config options added when you choose ``-m cori-haswell`` when setting up test
cases or a test suite:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/mesh_database
    initial_condition_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/initial_condition_database
    bathymetry_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /global/cfs/cdirs/e3sm/software/anaconda_envs/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 32

    # the number of multiprocessing or dask threads to use
    threads = 16

And here are the same for ``-m cori-knl``:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/mesh_database
    initial_condition_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/initial_condition_database
    bathymetry_database = /global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /global/cfs/cdirs/e3sm/software/anaconda_envs/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 68

    # the number of multiprocessing or dask threads to use
    threads = 18

cori, gnu
---------

.. code-block:: bash

    module switch PrgEnv-intel PrgEnv-gnu
    module load cray-netcdf-hdf5parallel
    module load cray-parallel-netcdf
    module load cmake
    source /global/project/projectdirs/e3sm/software/anaconda_envs/load_latest_e3sm_unified.sh
    export PIO=/global/u2/h/hgkang/my_programs/Scorpio
    git submodule update --init --recursive

    # debug:
    make gnu-nersc CORE=ocean USE_PIO2=true OPENMP=false DEBUG=true GEN_F90=true

    # optimized:
    make gnu-nersc CORE=ocean USE_PIO2=true OPENMP=false

cori, intel
-----------

.. code-block:: bash

    module rm intel
    module load intel/18.0.1.163
    module load cray-mpich/7.7.6
    module load cray-hdf5-parallel/1.10.2.0
    module load cray-netcdf-hdf5parallel/4.6.1.3
    module load cray-parallel-netcdf/1.8.1.4
    export PIO_VERSION=1.10.1
    export PIO=/global/homes/m/mpeterse/libraries/pio-${PIO_VERSION}-intel
    git submodule update --init --recursive

    make intel-nersc CORE=ocean

PIO on cori
-----------

We have already compiled PIO on cori, and paths are given in the previous
instructions. If you need to compile it yourself, you can do that as follows
(instructions from `xylar <http://github.com/xylar>`_).

.. code-block:: bash

    #!/bin/bash

    export PIO_VERSION=1.10.1

    rm -rf ParallelIO pio-${PIO_VERSION}

    git clone git@github.com:NCAR/ParallelIO.git
    cd ParallelIO
    git checkout pio$PIO_VERSION

    cd pio

    export PIOSRC=`pwd`
    git clone git@github.com:PARALLELIO/genf90.git bin
    git clone git@github.com:CESM-Development/CMake_Fortran_utils.git cmake
    cd ../..

    # Purge environment:
    module rm PrgEnv-cray
    module rm PrgEnv-gnu
    module rm PrgEnv-intel

    module load PrgEnv-intel/6.0.5
    module rm intel
    module load intel/18.0.1.163

    module rm craype
    module load craype/2.5.18

    module rm pmi
    module load pmi/5.0.14

    module rm cray-netcdf
    module rm cray-netcdf-hdf5parallel
    module rm cray-parallel-netcdf
    module rm cray-hdf5-parallel
    module rm cray-hdf5

    module rm cray-mpich
    module load cray-mpich/7.7.6

    # Load netcdf and pnetcdf modules
    module load cray-hdf5-parallel/1.10.2.0
    module load cray-netcdf-hdf5parallel/4.6.1.3
    module load cray-parallel-netcdf/1.8.1.4

    export NETCDF=$NETCDF_DIR
    export PNETCDF=$PARALLEL_NETCDF_DIR
    export PHDF5=$HDF5_DIR
    export MPIROOT=$MPICH_DIR

    export FC=ftn
    export CC=cc
    mkdir pio-${PIO_VERSION}
    cd pio-${PIO_VERSION}
    cmake -D NETCDF_C_DIR=$NETCDF -D NETCDF_Fortran_DIR=$NETCDF \
       -D PNETCDF_DIR=$PNETCDF -D CMAKE_VERBOSE_MAKEFILE=1 $PIOSRC
    make

    DEST=$HOME/libraries/pio-${PIO_VERSION}-intel
    rm -rf $DEST
    mkdir -p $DEST
    cp *.a *.h *.mod $DEST

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
