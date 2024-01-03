Test E3SM submodule changes
===========================

This utility is designed to test a sequence of E3SM commits related to a given
MPAS component between two commit hashes (assumed to be the current and new
hashes for the submodule).  The utility finds merges between the current and
the new hash that relate to the ocean and/or land-ice components and the MPAS
framework.  The `test_e3sm_changes.py` scrip manages the process with the
help of a config file similar to `example.cfg`.

Instructions
------------

1. Configure the compass environment and create load scripts with the desiredof
   compiler and mpi library, e.g.:
   ```shell
   ./conda/configure_compass_env.py --env_name compass_e3sm_update \
       --compiler intel --mpi openmpi --conda ~/miniforge3/
   ```

2. Copy `example.cfg` to the base of the branch:
   ```shell
   cp utils/e3sm_update/example.cfg e3sm_update.cfg
   ```

3. Modify the config options with the current and new hashes, the test cases
   or suite you wish to run, the base work directory, etc.

4. On a login node, run:
   ```shell
   ./utils/e3sm_update/test_e3sm_changes.py -f e3sm_update.cfg
   ```

5. Worktrees will be created for the current and new submodules as well as
   each relevant E3SM pull request inbetween.  The code will be built with the
   make command provided in the config file and then a job will be submitted
   for running the suite or test case(s) provided in the config file, comparing
   with the previous pull request of interest as a baseline.
