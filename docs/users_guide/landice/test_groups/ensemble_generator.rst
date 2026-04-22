.. _landice_ensemble_generator:

ensemble_generator
==================

The ``landice/ensemble_generator`` test group creates ensembles of MALI
simulations with different parameter values.  The ensemble framework
sets up a user-defined number of simulations with parameter values selected
from uniform sampling, log-uniform sampling, or a space-filling Sobol
sequence.

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

Parameter types
---------------

Parameters are defined in ``[ensemble.parameters]`` and fall into two
categories:

* ``special`` parameters: parameters without the ``nl.`` prefix that use
  custom setup logic beyond namelist replacement

* ``namelist`` parameters: parameters prefixed with ``nl.`` that map directly
  to one or more float namelist options through ``<name>.option_name``.
  Note that only float namelist options are currently supported, but the framework
  does not validate that the options defined in the config file are actually float
  namelist options.  Typically, ``<name>.option_name`` will indicate a single 
  namelist option, but it can indicate multiple options if the same parameter
  should be applied to multiple namelist options (e.g., for grounded and
  floating von Mises threshold stresses).

The currently supported special parameters are:

* ``fric_exp``: basal friction power-law exponent (requires modifying
  ``muFriction`` and ``albany_input.yaml``)

* ``mu_scale``: multiplicative scale factor for ``muFriction`` in the
  modified input file

* ``stiff_scale``: multiplicative scale factor for ``stiffnessFactor`` in the
  modified input file

* ``gamma0``: ISMIP6-AIS basal-melt sensitivity coefficient

* ``meltflux``: target ice-shelf basal melt flux, converted to ``deltaT``
  using ``gamma0`` and domain-mean thermal forcing

In addition, spinup runs can optionally apply precomputed friction samples
from Albany inversion products through ``[spinup_ensemble]`` config options
(``fric_samples_file``, ``fric_map_file``, ``mpas_cellid_file``, and
``fric_sample_scale``).  When these options are provided, run ``N`` uses
sample index ``N`` from ``fric_samples_file`` and writes the resulting
``muFriction`` field into that run's copied input file.

Test cases
----------

The test group includes two test cases:

* ``spinup_ensemble``: a set of simulations from the same initial condition
  but with different parameter values.  This could either be fixed climate
  relaxation spinup or forced by time-evolving historical conditions.

* ``branch_ensemble``: a set of simulations branched from each member of the
  spinup_ensemble in a specified year with a different forcing.  Multiple
  branch ensembles can be branched from one spinup_ensemble

Test case operations
--------------------

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

Ensemble templates
------------------

This test group uses a template-based configuration workflow.
Instead of maintaining one set of test-group resource files, each model
configuration lives in its own subdirectory under
``ensemble_templates/<name>`` with separate spinup and branch
cfg/namelist/streams resources.  Users typically select a template via the
``[ensemble_generator] ensemble_template`` option or create a new template.
The user may also provide custom overrides in a user cfg file.
A new ensemble template should be added for each new study by creating
a new subdirectory under ``ensemble_templates/`` with the same structure as
the existing templates and following a naming convention like:
``<domain_topic_year>``, e.g., ``amery4km_probproj_2024`` or
``ais4km_hydro_2026``.

The selected template controls which config files and model resource files are
used for the spinup and branch cases.  The package layout is:

.. code-block:: none

   compass/landice/tests/ensemble_generator/ensemble_templates/<name>/
     spinup/
       ensemble_generator.cfg
       namelist.landice
       streams.landice
       albany_input.yaml
     branch/
       branch_ensemble.cfg
       namelist.landice
       streams.landice

config options
--------------

The shared config option for this test group is:

.. code-block:: cfg

   [ensemble_generator]

   # name of the ensemble template to use
   # resources are loaded from:
   # compass.landice.tests.ensemble_generator.ensemble_templates.<name>
   ensemble_template = amery4km_probproj_2024

The template-specific spinup config options (from
``ensemble_templates/<name>/spinup/ensemble_generator.cfg``) are:

.. code-block:: cfg

   [ensemble_generator]

   # start and end numbers for runs to set up and run
   # Run numbers should be zero-based.
   # Additional runs can be added and run to an existing ensemble
   # without affecting existing runs, but trying to set up a run
   # that already exists will generate a warning and skip that run.
   # If using uniform or log-uniform sampling, start_run should be 0 and
   # end_run should be equal to (max_samples - 1), otherwise unexpected
   # behavior may result.
   # These values do not affect viz/analysis, which will include any
   # runs it finds.
   start_run = 0
   end_run = 3

   # sampling_method can be 'sobol' for a space-filling Sobol sequence,
   # 'uniform' for linear sampling, or 'log-uniform' for logarithmic
   # sampling between min and max parameter bounds.
   # Uniform and log-uniform are most appropriate for a single-parameter
   # sensitivity study because they sample each active parameter using the
   # same rank ordering, thus sampling only a small fraction of parameter
   # space in higher dimensions.
   sampling_method = sobol

   # maximum number of samples to be considered.
   # max_samples needs to be greater or equal to (end_run + 1)
   # When using uniform or log-uniform sampling, max_samples should equal
   # (end_run + 1).
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

   # number of tasks that each ensemble member should be run with
   # Eventually, compass could determine this, but we want explicit control for now
   ntasks = 128

    [spinup_ensemble]

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

    # Optional friction-sample inputs from Albany optimization products.
    # When these options are present, run N uses sample index N.
    # fric_samples_file = /path/to/postVarSamples-5000.npy
    # fric_map_file = /path/to/mu_log_opt.ascii
    # mpas_cellid_file = /path/to/mpas_cellID.ascii
    # fric_sample_scale = 0.25

    # For meltflux perturbations, this observed ice-shelf area is used when
    # converting target melt flux to deltaT.
    iceshelf_area_obs = 60654.e6

The parameter sampling definitions live in a separate section,
``[ensemble.parameters]``.  The order listed sets the sampling
dimension ordering, special parameters are unprefixed, and namelist
parameters use the ``nl.`` prefix with a companion ``.option_name``.

For ``log-uniform`` sampling, each parameter bound must be strictly
positive because sampling is performed in log space.

.. code-block:: cfg

   [ensemble.parameters]

   # special parameters (handled by custom code)
   fric_exp = 0.1, 0.33333
   mu_scale = 0.8, 1.2
   stiff_scale = 0.8, 1.2
   gamma0 = 9620.0, 471000.0
   meltflux = 12.0, 58.0

   # namelist float parameters (generic handling)
   nl.von_mises_threshold = 80.0e3, 180.0e3
   nl.von_mises_threshold.option_name =
     config_grounded_von_Mises_threshold_stress,
     config_floating_von_Mises_threshold_stress

   nl.calv_spd_limit = 0.0001585, 0.001585
   nl.calv_spd_limit.option_name = config_calving_speed_limit

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
can be applied to any domain and set of input files using the ensemble templates
discussed above.

The initial condition and forcing files are specified in the selected
template file
``compass/landice/tests/ensemble_generator/ensemble_templates/<name>/spinup/ensemble_generator.cfg``

branch_ensemble
---------------

``landice/ensemble_generator/branch_ensemble`` uses the ensemble framework to create
an ensemble of simulations that are branched from corresponding runs of the
``spinup_ensemble`` at a specified year with a different forcing.  In general,
any namelist or streams modifications can be applied to the branch runs.

The branch_ensemble config options are read from the selected template file
``compass/landice/tests/ensemble_generator/ensemble_templates/<name>/branch/branch_ensemble.cfg``.

Steps for setting up and running an ensemble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. With a compass conda environment set up, run, e.g.,
   ``compass setup -t landice/ensemble_generator/spinup_ensemble -w WORK_DIR_PATH -f TEMPLATE.cfg``
   where ``WORK_DIR_PATH`` is a location that can store the whole
   ensemble (typically a scratch drive) and ``TEMPLATE.cfg`` is the
   config for the ensemble template to be set up, i.e., located at:
   ``landice/tests/ensemble_generator/ensemble_templates/TEMPLATE_NAME/spinup/ensemble_generator.cfg``
   The cfg file should include options for ``[parallel]`` and ``[job]``.
   The cfg files for each template should in general be committed in a
   state that is ready to use, but users should review the contents
   before setting up an ensemble, including the ``start_run`` and
   ``end_run`` values.  If adjustments are needed, the user can make
   adjustments to the cfg file in place, or make a copy and point to
   that directly.

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
