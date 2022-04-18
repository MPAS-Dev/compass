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
   This will save some info in `conda/logs/matrix.log` that is needed for the
   remainder of the build.

2. Copy `example.cfg` to the base of the branch:
   ```shell
   cp utils/matrix/example.cfg matrix.cfg
   ```

3. Modify the config options with the appropriate compilers and MPI libraries;
   whether to build in debug mode, optimized or both; and whether to build with
   OpenMPI or not (or both)

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

8. The matrix build doesn't take care of running the jobs on a compute node.
   You will need to do that yourself.
