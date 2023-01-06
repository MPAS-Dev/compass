Matrix build and setup
======================

Often, developers want to test a given set of test cases or a test suite with
multiple compilers and configurations.  Doing this process manually can be 
challenging and prone to mistakes.  The `setup_matrix.py` scrip is designed
to automate this process with the help of a config file similar to 
`example.cfg`.

Instructions
------------

1. Configure the compass environment and create load scripts with a matrix of
   compilers and mpi libraries as desired, e.g.:
   ```shell
   ./conda/configure_compass_env.py --env_name compass_matrix \
       --compiler all --mpi all --conda ~/miniconda3/
   ```
   See the next section for more details.  This will save some info in 
   `conda/logs/matrix.log` that is needed for the remainder of the build.  If
   you want to set up load scripts for more environments than you actually
   want to run your matrix on, you can edit `matrix.log`.  The syntax is:
   ```
   <machine>
   <compiler>, <mpi>
   <compiler>, <mpi>
   <compiler>, <mpi>
   ...
   ```
   It is also safe to rerun `./conda/configure_compass_env.py` with just the
   subset of compilers and MPI libraries you want for the matrix.  Any
   load scripts created in previous calls will not be deleted and the conda
   environment will just be updated, not recreated unless you explicitly ask
   for it to be recreated.

2. Copy `example.cfg` to the base of the branch:
   ```shell
   cp utils/matrix/example.cfg matrix.cfg
   ```

3. Modify the config options with the appropriate compilers and MPI libraries;
   whether to build in debug mode, optimized or both; and whether to build with
   OpenMP or not (or both)

4. Set the conda environment name (or prefix on Linux and OSX) that you want to
   use

5. Modify the various paths and commands as needed.

6. Add any other config options you want to pass on to the `compass setup` or
   `compass suite` command.  A config file will be written out for each
   build configuration you select.

7. On a login node, run:
   ```shell
   ./utils/matrix/setup_matrix.py -f matrix.cfg
   ```
   Optionally use the `--submit` flag to submit jobs once each configuration
   has been built and set up.

8. If you do not use the `--submit` flag, the matrix build doesn't take care of
   running the jobs on a compute node.  You will need to do that yourself.

Matrix of compilers and MPI libraries
-------------------------------------

You control the matrix of compilers and MPI libraries by how you call
```shell
./conda/configure_compass_env.py ...
```
This is because the configure script knows about which compilers and MPI 
libraries are  available for a given machine, something tricky to figure out in 
`./util/matrix/setup_matrix.py` directly.

You can do:
```shell
./conda/configure_compass_env.py --compiler all --mpi all ...
```
to get all supported compilers and MPI libraries.  You can do:
```shell
./conda/configure_compass_env.py --compiler intel --mpi all ...
```
to get just the `intel` configurations.  You can do:
```shell
./conda/configure_compass_env.py --compiler all --mpi openmpi ...
```
to get just the `openmpi` configurations or:
```shell
./conda/configure_compass_env.py --compiler intel ...
```
(omitting the `--mpi`) to get the default MPI library for intel.  Finally, you
can give a list of compilers and a list of the same length of MPI libraries,
such as:
```shell
./conda/configure_compass_env.py --compiler intel intel intel gnu \
    --mpi impi openmpi mvapich mvapich ...
```
This will give you 3 intel variants and a gnu variant.
