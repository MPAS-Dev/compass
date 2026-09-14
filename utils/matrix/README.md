Matrix build and setup
======================

Often, developers want to test a given set of test cases or a test suite with
multiple compilers and configurations.  Doing this process manually can be
challenging and prone to mistakes.  The `setup_matrix.py` scrip is designed
to automate this process with the help of a config file similar to
`example.cfg`.

**Note:** `setup_matrix.py` has not yet been updated for `./deploy.py`.  It
still expects the `conda/logs/matrix.log` file and the
`load_<env_name>_<machine>_<compiler>_<mpi>.sh` load-script names that the
old `./conda/configure_compass_env.py` produced.  `./deploy.py` writes neither,
so the script needs to be updated before the instructions below will work.

Instructions
------------

1. Deploy the compass environment and create load scripts with a matrix of
   compilers and MPI libraries as desired, e.g.:
   ```shell
   ./deploy.py --compiler intel gnu --mpi openmpi openmpi
   ```
   See the next section for more details.  It is safe to rerun `./deploy.py`
   with just the subset of compilers and MPI libraries you want for the
   matrix.  Any load scripts created in previous calls will not be deleted and
   the pixi environment will just be updated, not recreated unless you
   explicitly ask for it to be recreated with `--recreate`.

2. Copy `example.cfg` to the base of the branch:
   ```shell
   cp utils/matrix/example.cfg matrix.cfg
   ```

3. Modify the config options with the appropriate compilers and MPI libraries;
   whether to build in debug mode, optimized or both; and whether to build with
   OpenMP or not (or both)

4. Modify the various paths and commands as needed.

5. Add any other config options you want to pass on to the `compass setup` or
   `compass suite` command.  A config file will be written out for each
   build configuration you select.

6. On a login node, run:
   ```shell
   ./utils/matrix/setup_matrix.py -f matrix.cfg
   ```
   Optionally use the `--submit` flag to submit jobs once each configuration
   has been built and set up.

7. If you do not use the `--submit` flag, the matrix build doesn't take care of
   running the jobs on a compute node.  You will need to do that yourself.

Matrix of compilers and MPI libraries
-------------------------------------

You control the matrix of compilers and MPI libraries by how you call
```shell
./deploy.py ...
```
This is because the deployment knows about which compilers and MPI
libraries are available for a given machine, something tricky to figure out in
`./util/matrix/setup_matrix.py` directly.

You can do:
```shell
./deploy.py --compiler intel ...
```
(omitting the `--mpi`) to get the default MPI library for intel.  You can
give a list of compilers and a list of the same length of MPI libraries,
such as:
```shell
./deploy.py --compiler intel gnu --mpi openmpi openmpi ...
```
This will give you an intel variant and a gnu variant, both with `openmpi`.
If one of the lists has a single entry, it is used for every entry in the
other list, so
```shell
./deploy.py --compiler intel gnu --mpi openmpi ...
```
is equivalent.
