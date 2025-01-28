.. _landice_ismip6_run:

ismip6_run
==========

The ``landice/ismip6_run`` test group includes a test case for setting up
one or more experiments from the `ISMIP6 set of experiments <https://www.climate-cryosphere.org/wiki/index.php?title=ISMIP6_wiki_page>`_.
This functionality assumes the initial condition, parameter, and forcing files
have already been generated using the :ref:`landice_ismip6_forcing` test case.
It creates a consistent set of run directories
for the experiments requested.  It is not meant for automated running of the
experiments, and expert knowledge is recommended for conducting the actual
experiments.

At present, there is a single test case for the
ISMIP6-Projections2300-Antarctica ensemble.

ismip6_ais_proj2300
-------------------

``landice/ismip6_run/ismip6_ais_proj2300`` sets up one or more experiments
from the protocol for
`ISMIP6-Projections2300-Antarctica <https://www.climate-cryosphere.org/wiki/index.php?title=ISMIP6-Projections2300-Antarctica>`_.

The user should review and modify all values in the default config for
specifying input files and model information.  See config comments for
detailed information about each option.
Note that default filepaths refer to files at NERSC and may need to be updated
depending on what machine is being used and user permissions.
It may also be necessary to
alter namelist, streams, or test case code if non-standard configurations
are to be used.

config options
--------------

All config options should be reviewed and altered as needed.
Example cfg files are included in the test case directory within the
``compass`` repo for previously used configurations on specific
machines to facilitate reproducibility and form a useful starting place
for new runs

.. code-block:: cfg

    [ismip6_run_ais_2300]

    # list of experiments to set up.
    # Can be "tier1", "tier2", "all", or a comma-delimited list of runs
    # "tier1" is expAE01-expAE06 plus hist and ctrlAE
    # "tier2" is expAE07-expAE14
    exp_list = tier1

    # Resolution that will be set up, in km.  Should be one of 4, 8, as those are
    # the two resolutions currently supported.
    # mesh_res is informational only and is used for directory naming conventions
    # The filepaths below must manually be set to be consistent.
    mesh_res = 8

    # number of tasks to use for each run
    ntasks = 128

    # value to use for config_pio_stride.
    # Should be divisible into ntasks
    pio_stride = 128

    # Base path to the processed input ismip6 ocean and smb forcing files.
    # User has to supply.
    forcing_basepath = /global/cfs/cdirs/fanssie/MALI_projects/ISMIP6-2300/forcing/ais_mesh_8to30km_res

    # Path to the initial condition file. User has to supply.
    init_cond_path = /global/cfs/cdirs/fanssie/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/relaxation_0TGmelt_10yr_muCap/AIS_8to30km_r01_20220906.smooth3.basinsFineTuned_carvedRonne_CIRWIP_relaxation_0TGmelt_10yr_muCap.nc

    # Path to the file for the basal melt parametrization coefficients.
    melt_params_path = /global/cfs/cdirs/fanssie/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/basin_and_coeff_gamma0_DeltaT_quadratic_non_local.nc

    # Path to the region mask file
    region_mask_path = /global/cfs/cdirs/fanssie/MALI_projects/ISMIP6-2300/initial_conditions/AIS_8to30km_20221027/AIS_8to30km_r01_20220607.regionMask_ismip6.nc

    # Calving method to use.  Should be one of:
    # 'restore' for restore-calving front method (fixed calving front)
    # 'von_mises' for von Mises threshold stress calving law
    calving_method = restore

    # Path to the file containing the von Mises parameter fields
    # groundedVonMisesThresholdStress and floatingVonMisesThresholdStress.
    # Only required if calving_method is set to 'von_mises'
    von_mises_parameter_path = UNKNOWN

    # Whether facemelting should be included in the runs
    include_face_melting = False

    # True if running coupled MALI-sea level model simulation
    sea_level_model = True

    # NOTE: for the directories related to the sea-level model input/outputs, the slash '/' at the end of the directory name is necessary.
    # Path to the base directory containing globally defined ice thickness field for the sea-level model
    slm_input_ice = /global/cfs/cdirs/fanssie/MALI_projects/SeaLevelModel_Inputs/icemodel/

    # Path to the directory containing earth model for the sea-level model
    slm_input_earth = /global/cfs/cdirs/fanssie/MALI_projects/SeaLevelModel_Inputs/earthmodel/

    # Earth structure profile
    # possible values: any file name string of earth model file.
    # Note that there is no single representative earth model for the globe or
    # any region of the globe (e.g., West Antarctica, East Antarctica, Greenland)
    # But for this test, we use 'prem_512.l60K2C.sum18p6.dum19p2.tz19p4.lm22' for West Antarctica
    # and 'prem_coll_512.l120C.ump5.lm10' for East Antarctica. Other earthmodels that can be used
    # should exist in the path defined in the config option 'slm_input_earth'.
    slm_earth_structure = prem_512.l60K2C.sum18p6.dum19p2.tz19p4.lm22

    # Path to the diectory containing other input files (present-day global topography and sea-level model grid files)
    slm_input_others = /global/cfs/cdirs/fanssie/MALI_projects/SeaLevelModel_Inputs/others/

    # number of gauss-legendre nodes in latitude (typically an integer multiple of 512)
    nglv = 2048

Additionally, a user should also include the following options (and possibly
others) that will be used for submitting the jobs for each ensemble member
(set to appropriate values for their usage situation):

.. code-block:: cfg

    [parallel]
    account = ALLOCATION_NAME_HERE
    qos = regular

    [job]
    wall_time = 10:00:00

Steps for setting up and running experiments
--------------------------------------------

1. With a compass conda environment set up, run, e.g.,
   ``compass setup -t landice/ismip6_run/ismip6_ais_proj2300 -w WORK_DIR_PATH -f USER.cfg``
   where ``WORK_DIR_PATH`` is a location that can store the whole
   ensemble (typically a scratch drive) and ``USER.cfg`` is the
   user-defined config described in the previous section that includes
   options for ``[parallel]`` and ``[job]``, as well as any required
   modifications to the ``[ismip6_run_ais_2300]`` section.  Likely, most or all
   attributes in the ``[ismip6_run_ais_2300]`` section need to be customized for a
   given application.  It is possible to set up the test case without the
   ``-f`` option, but generally users will need to make their own
   adjustments to the example syntax above include it.  Also, if you
   do not compile MALI in the ``MALI-Dev`` submodule within compass, you will
   need to include the ``-p`` option specifying the path to where you compiled
   MALI.

2. After ``compass setup`` completes and all runs are set up, go to the
   ``WORK_DIR_PATH`` and change to the
   ``landice/ismip6_run/ismip6_ais_proj2300`` subdirectory.
   From there you will see subdirectories for each experiment.

3. Each experiment is to be run individually.  Change to the subdirectory
   of the experiment you would like to run.  It is suggested you review the
   job script, namelist, and streams files to be sure everything is set as
   expected.  Then use ``sbatch`` to submit the job script.

Note that the ``hist`` run must be completed before any of the other
experiments can be run.  A symlink to the ``hist`` restart file from year
2015 exists in each of the other experiment subdirectories.

Important notes for running coupled MALI-SLM simulations
--------------------------------------------------------

Projection handling
~~~~~~~~~~~~~~~~~~~

Currently, MALI uses a planar mesh that projects the south pole with
polar stereographic projection of ellipsoidal Earth, and the sea-level model
uses a global grid of spherical Earth. This inconsistency in the Earth's
assumed shape (ellipsoid vs. sphere) in MALI and the SLM causes discrepancies
in the comparison of post-simulation ice mass calculations from the model outputs.
To address this issue, an interim solution has been made to project the southern
polar region onto the MALI planar mesh assuming a sphere (this is done by setting
the lat/long in the MALI mesh using the 'aid-bedmap2-sphere' projection string in
calling the function in 'set_lat_lon_fields_in_planar_grid.py'. Thus, the resulting
MALI mesh for coupled MALI-SLM simulations that are setup from this testgroup have
the lat/long values based off sphere. Once the simulation outputs are generated,
it is necessary to calculate and apply the scaling factors to the MALI outputs
to correct for the areal distortion arose from projecting the south pole on a sphere
onto plane. Only after applying the scaling factor, the MALI outputs will be
comparable to the SLM outputs. The equations to calculate the scaling factors are shown in
Eqn. 21-4 in https://pubs.usgs.gov/pp/1395/report.pdf
An example of the calculation for the MALI-SLM case can also be found in
the ``compass`` testgroup/testcase/ ``compass/landice/test/slm/circ_icesheet/``,
``visualize`` step (https://github.com/MPAS-Dev/compass/pull/748).

Mapping files
~~~~~~~~~~~~~

Coupling between the Sea-Level Model and MALI requires the generation of mapping files
before simulations are run.  When ``sea_level_model=True`` in the cfg file, an
additional test case step is created named ``mapping_files``.  Before any MALI
simulations are started, the user should run this step.  It will generate the required
mapping files, and they will be available in each experiment directory through symlinks.
Once the mapping files have been generated, the user can proceed to running the ``hist``
experiment.

Restarts
~~~~~~~~

Additionally, for restarts with the SLM to work correctly, the entire history of the
``OUTPUT_SLM`` and ``ICELOAD_SLM`` directories must be present.  Because the projection
experiments (ctrl and exp*) are branched off the hist run as restarts, this means these
two directories from the hist run must be manually copied to each projection run before
beginning it. There is not an easy way for this to happen automatically, so this step
must be done manually.
