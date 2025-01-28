.. _landice_ensemble_generator:

ensemble_generator
==================

The ``landice/ensemble_generator`` test group creates ensembles of MALI
simulations with different parameter values.  The ensemble framework
sets up a user-defined number of simulations with parameter values selected
from either uniform sampling or a space-filling Sobol sequence.

A test case in this test group consists of a number of ensemble members,
and one ensemble manager.
Each ensemble member is a step of the test case, and can be run separately
or as part of the complete ensemble.  Ensemble members are identified by a
three digit run number, starting with 000.
A config file specifies the run numbers to set up, as well as some common
information about the run configuration.

The test case can be generated multiple times to set up and run additional
runs with a different range of run numbers after being run initially. This
allows one to perform a small ensemble (e.g. 2-10 runs) to make sure results
look as expected before spending time on a larger ensemble. This also allows
one to add more ensemble members from the Sobol sequence later if UQ analysis
indicates the original sample size was insufficient.

A number of possible parameters are supported and whether they are active and
what parameter value ranges should be used are specified in a user-supplied
config file.  Currently these parameters are supported:

* basal friction power law exponent

* scaling factor on muFriction

* scaling factor on stiffnessFactor

* von Mises threshold stress for calving

* calving rate speed limit

* gamma0 melt sensitivity parameter in ISMIP6-AIS ice-shelf basal melting
  parameterization

* target ice-shelf basal melt rate for ISMIP6-AIS ice-shelf basal melting
  parameterization.  In the model setup, the deltaT thermal forcing bias
  adjustment is adjusted to obtain the target melt rate for a given gamma0

Additional parameters can be easily added in the future.

``compass setup`` will set up the simulations and the ensemble manager.
``compass run`` from the test case work directory will submit each run as a
separate slurm job.
Individual runs can be run independently through the job script or
``compass run`` executed in the
run directory.  (E.g., if you want to test or debug a run without running the
entire ensemble.)

Simulation output can be analyzed with the ``plot_ensemble.py`` visualization
script, which generates plots of basic quantities of interest as a function
of parameter values, as well as identifies runs that did not reach the
target year.  The visualization script plots a small number of quantities of
interest as a function of each active parameter.  It also plots pairwise
parameter sensitivities for each pair of parameters being varied.  Finally,
it plots time-series plots for the quantities of interest for all runs in the
ensemble.  There are a number of options in the visualization script that
can be manually set near the top of script.

Future improvements may include:

* enabling the ensemble manager to identify runs that need to be restarted
  so the restarts in the spinup_ensemble do not need to be managed manually

* safety checks or warnings before submitting ensembles that will use large
  amounts of computing resources

* a method for maintaining namelist, streams, and albany_input.yaml files for
  different ensembles.  Currently, these input files are specific to the Amery
  Ice Shelf ensemble run in 2023.

The test group includes two test cases:

* ``spinup_ensemble``: a set of simulations from the same initial condition
  but with different parameter values.  This could either be fixed climate
  relaxation spinup or forced by time-evolving historical conditions.

* ``branch_ensemble``: a set of simulations branched from each member of the
  spinup_ensemble in a specified year with a different forcing.  Multiple
  branch ensembles can be branched from one spinup_ensemble

config options
--------------
Test cases in this test group have the following common config options.

This test group is intended for expert users, and it is expected that it
will typically be run with a customized cfg file.  Note the default run
numbers create a small ensemble, but uncertainty quantification applications
will typically need dozens or more simulations.

The test-case-specific config options are:

.. code-block:: cfg

   [ensemble]

   # start and end numbers for runs to set up and run
   # Run numbers should be zero-based.
   # Additional runs can be added and run to an existing ensemble
   # without affecting existing runs, but trying to set up a run
   # that already exists will generate a warning and skip that run.
   # If using uniform sampling, start_run should be 0 and end_run should be
   # equal to (max_samples - 1), otherwise unexpected behavior may result.
   # These values do not affect viz/analysis, which will include any
   # runs it finds.
   start_run = 0
   end_run = 3

   # sampling_method can be either 'sobol' for a space-filling Sobol sequence
   # or 'uniform' for uniform sampling.  Uniform sampling is most appropriate
   # for a single parameter sensitivity study.  It will sample uniformly across
   # all dimensions simultaneously, thus sampling only a small fraction of
   # parameter space
   sampling_method = sobol

   # maximum number of samples to be considered.
   # max_samples needs to be greater or equal to (end_run + 1)
   # When using uniform sampling, max_samples should equal (end_run + 1).
   # When using Sobol sequence, max_samples ought to be a power of 2.
   # max_samples should not be changed after the first set of ensemble.
   # So, when using Sobol sequence, max_samples might be set larger than
   # (end_run + 1) if you plan to add more samples to the ensemble later.
   max_samples = 1024

   # basin for comparing model results with observational estimates in
   # visualization script.
   # Basin options are defined in compass/landice/ais_observations.py
   # If desired basin does not exist, it can be added to that dataset.
   # (They need not be mutually exclusive.)
   # If a basin is not provided, observational comparisons will not be made.
   basin = ISMIP6BasinBC

   # fraction of CFL-limited time step to be used by the adaptive timestepper
   # This value is explicitly included here to force the user to consciously
   # select the value to use.  Model run time tends to be inversely proportional
   # to scaling this value (e.g., 0.2 will be ~4x more expensive than 0.8).
   # Value should be less than or equal to 1.0, and values greater than 0.9 are
   # not recommended.
   # Values of 0.7-0.9 typically work for most simulations, but some runs may
   # fail.  Values of 0.2-0.5 are more conservative and will allow more runs
   # to succeed, but will result in substantially more expensive runs
   # However, because the range of parameter combinations being simulated
   # are likely to stress the model, a smaller number than usual may be
   # necessary to effectively cover parameter space.
   # A user may want to do a few small ensembles with different values
   # to inform the choice for a large production ensemble.
   cfl_fraction = 0.7

   # Path to the initial condition input file.
   # Eventually this could be hard-coded to use files on the input data
   # server, but initially we want flexibility to experiment with different
   # inputs and forcings
   input_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/Amery.nc

   # the value of the friction exponent used for the calculation of muFriction
   # in the input file
   orig_fric_exp = 0.2

   # Path to ISMIP6 ice-shelf basal melt parameter input file.
   basal_melt_param_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/forcing/basal_melt/parameterizations/Amery_4to20km_basin_and_coeff_gamma0_DeltaT_quadratic_non_local_median_allBasin2.nc

   # Path to thermal forcing file for the mesh to be used
   TF_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/forcing/ocean_thermal_forcing/obs/Amery_4to20km_obs_TF_1995-2017_8km_x_60m.nc

   # Path to SMB forcing file for the mesh to be used
   SMB_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/forcing/atmosphere_forcing/RACMO_climatology_1995-2017/Amery_4to20km_RACMO2.3p2_ANT27_smb_climatology_1995-2017_no_xtime_noBareLandAdvance.nc

   # number of tasks that each ensemble member should be run with
   # Eventually, compass could determine this, but we want explicit control for now
   ntasks = 128

   # whether basal friction exponent is being varied
   # [unitless]
   use_fric_exp = True
   # min value to vary over
   fric_exp_min = 0.1
   # max value to vary over
   fric_exp_max = 0.33333

   # whether a scaling factor on muFriction is being varied
   # [unitless: 1.0=no scaling]
   use_mu_scale = True
   # min value to vary over
   mu_scale_min = 0.8
   # max value to vary over
   mu_scale_max = 1.2

   # whether a scaling factor on stiffnessFactor is being varied
   # [unitless: 1.0=no scaling]
   use_stiff_scale = True
   # min value to vary over
   stiff_scale_min = 0.8
   # max value to vary over
   stiff_scale_max = 1.2

   # whether the von Mises threshold stress (sigma_max) is being varied
   # [units: Pa]
   use_von_mises_threshold = True
   # min value to vary over
   von_mises_threshold_min = 80.0e3
   # max value to vary over
   von_mises_threshold_max = 180.0e3

   # whether the calving speed limit is being varied
   # [units: km/yr]
   use_calv_limit = False
   # min value to vary over
   calv_limit_min = 5.0
   # max value to vary over
   calv_limit_max = 50.0

   # whether ocean melt parameterization coefficient is being varied
   # [units: m/yr]
   use_gamma0 = True
   # min value to vary over
   gamma0_min = 9620.0
   # max value to vary over
   gamma0_max = 471000.0

   # whether target ice-shelf basal melt flux is being varied
   # [units: Gt/yr]
   use_meltflux = True
   # min value to vary over
   meltflux_min = 12.
   # max value to vary over
   meltflux_max = 58.
   # ice-shelf area associated with target melt rates
   # [units: m^2]
   iceshelf_area_obs = 60654.e6

A user should copy the default config file to a user-defined config file
before setting up the test case and any necessary adjustments made.
Importantly, the user-defined config should be modified
to also include the following options that will be used for submitting the
jobs for each ensemble member.

.. code-block:: cfg

   [parallel]
   account = ALLOCATION_NAME_HERE
   qos = regular

   [job]
   wall_time = 1:30:00

Note that currently there is not functionality
to automatically enable restart settings if runs in the spinup_ensemble
do not reach the desired year.  This could be added in the future, but to
date it has been practical to set ``wall_time`` long enough to ensure this
is not a problem.  Runs in a branch_ensemble are set as restarts from the
spinup_ensemble runs, so there is no need to change settings if runs
need to be continued beyond the first job.

spinup_ensemble
---------------

``landice/ensemble_generator/spinup_ensemble`` uses the ensemble framework to create
an ensemble of simulations integrated over a specified time range.  The test case
can be applied to any domain and set of input files.  If the default namelist
and streams settings are not appropriate, they can be adjusted or a new test
case can be set up mirroring the existing one.

The default model configuration uses:

* first-order velocity solver

* power law basal friction

* evolving temperature

* von Mises calving

* ISMIP6 surface mass balance and sub-ice-shelf melting using climatological
  mean forcing

The initial condition and forcing files are specified in the
``ensemble_generator.cfg`` file or a user modification of it.

branch_ensemble
---------------

``landice/ensemble_generator/branch_ensemble`` uses the ensemble framework to create
an ensemble of simulations that are branched from corresponding runs of the
``spinup_ensemble`` at a specified year with a different forcing.  In general,
any namelist or streams modifications can be applied to the branch runs.

The branch_ensemble test-case-specific config options are:

.. code-block:: cfg

   # config options for setting up an ensemble

   # config options for branching an ensemble
   [branch_ensemble]

   # start and end numbers for runs to set up and run
   # branch runs.
   # It is assumed that spinup runs have already been
   # conducted for these runs.
   start_run = 0
   end_run = 3

   # Path to thermal forcing file for the mesh to be used in the branch run
   TF_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/forcing/ocean_thermal_forcing/UKESM1-0-LL_SSP585/1995-2300/Amery_4to20km_TF_UKESM1-0-LL_SSP585_2300.nc

   # Path to SMB forcing file for the mesh to be used in the branch run
   SMB_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Amery_UQ/Amery_4to20km_from_whole_AIS/forcing/atmosphere_forcing/UKESM1-0-LL_SSP585/1995-2300/Amery_4to20km_SMB_UKESM1-0-LL_SSP585_2300_noBareLandAdvance.nc

   # location of spinup ensemble to branch from
   spinup_test_dir = /pscratch/sd/h/hoffman2/AMERY_corrected_forcing_6param_ensemble_2023-03-18/landice/ensemble_generator/ensemble

   # year of spinup simulation from which to branch runs
   branch_year = 2050

   # whether to only set up branch runs for filtered runs or all runs
   set_up_filtered_only = True

   # path to pickle file containing filtering information generated by plot_ensemble.py
   ensemble_pickle_file = None

Steps for setting up and running an ensmble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. With a compass conda environment set up, run, e.g.,
   ``compass setup -t landice/ensemble_generator/spinup_ensemble -w WORK_DIR_PATH -f USER.cfg``
   where ``WORK_DIR_PATH`` is a location that can store the whole
   ensemble (typically a scratch drive) and ``USER.cfg`` is the
   user-defined config described in the previous section that includes
   options for ``[parallel]`` and ``[job]``, as well as any required
   modifications to the ``[ensemble]`` section.  Likely, most or all
   attributes in the ``[ensemble]`` section need to be customized for a
   given application.

2. After ``compass setup`` completes and all runs are set up, go to the
   ``WORK_DIR_PATH`` and change to the
   ``landice/ensemble_generator/spinup_ensemble`` subdirectory.
   From there you will see subdirectories for each run, a subdirectory for the
   ``ensemble_manager`` and symlink to the visualization script.

3. To submit jobs for the entire ensemble, change to the ``ensemble_manager``
   subdirectory and execute ``compass run``.  Be careful, as it is possible to
   consume a large number of computing resources quickly with this tool!

4. Each run will have its own batch job that can be monitored with ``squeue``
   or similar commands.

5. When the ensemble has completed, or as it is progressing,
   you can assess the result through the
   basic visualization script ``plot_ensemble.py``.  The script will skip runs
   that are incomplete or failed, so you can run it while an ensemble is
   still running to assess progress.

6. If you want to run add additional ensemble members, adjust
   ``start_run`` and ``end_run`` in your config file and redo steps 1-5.
   The ensemble_manager will always be set to run the most recent run
   numbers defined in the config when ``compass setup`` was run.
   The visualization script is independent of the run manager and will
   process all runs it finds.

It is also possible to run an individual run manually by changing to the run
directory and submitting the job script yourself with ``sbatch``.

Setting up and running a branch ensemble follows the same steps.  Multiple
branch ensembles (e.g., with different climate forcing scenarios) can be
conducted from one spinup ensemble.

It is intended that a single MALI executable is used for all simulations in an
ensemble.  If the version of the code is updated, it is up to the user to anticipate
undesirable consequences.
