.. _landice_ismip7_calibration:

ismip7_calibration
==================

The ``landice/ismip7_calibration`` test group calibrates MALI's sub-shelf
melt parameterization against the Ice Sheet Model Intercomparison for CMIP7
(ISMIP7) Antarctic ice-ocean protocol (Reese et al., Sect. 4.2), and produces
the 5th, 50th and 95th percentiles of the melt parameter that the ISMIP7 MALI
projections need.

The protocol fits the free parameter of the melt module against four terms:

===== ==========================================================
term  what it constrains
===== ==========================================================
J1    present-day melt integrated over each IMBIE2 drainage basin
J2    present-day melt integrated over bins of equal buttressing importance
J3    the warm-minus-cold basin-mean melt difference of ocean models
J4    observed melt of Pine Island and Dotson, per observation year
===== ==========================================================

It then draws 100,000 random samples of the term weights and of the targets
within their uncertainties, minimising the objective for each draw.  The
distribution of the minimisers gives the percentiles.

The test group includes two test cases.

* ``replication`` reproduces the published 8 km calibration through compass's
  own code path.  It runs in about a minute on one core, needs no MALI run,
  and needs only the ISMIP7 datasets.  Use it to check that everything is
  wired up before running the full calibration.

* ``ais`` is the calibration itself on an Antarctic MALI mesh: it remaps the
  ISMIP7 masks and forcing, runs one single-timestep MALI melt diagnostic per
  ocean state, aggregates the melt, selects the parameter and fits the
  per-basin thermal-forcing correction ``dT_b``.

Two melt forms can be calibrated, and by default both are:

``ismip7``
    the Burgard et al. (2022) **local** quadratic recommended for ISMIP7,
    calibrating ``K``

``ismip6``
    the **non-local** quadratic MALI already had, calibrating ``gamma0``.
    With a constant salinity this is algebraically the same as the Burgard
    *semi-local* form, so this is how that form is calibrated.

.. _landice_ismip7_calibration_salinity:

A note on salinity
------------------

The calibration runs MALI with a **constant** ocean salinity.  MALI can read
a 3-D salinity field, but the ISMIP7 ocean forcing already processed for the
MALI projections carries thermal forcing only, so a projection has no
salinity field to read.  Calibrating against a spatially varying salinity
would tune the melt parameter for physics the projections cannot run.

Melt is linear in salinity, and basin-mean salinity spans about 34.1 to 34.7
against the default constant of 34.5, so this shifts the calibrated parameter
by roughly 1%.  The published ``K`` was obtained with a locally varying
salinity, so comparison with it is approximate at about that level.

.. _landice_ismip7_calibration_usage:

Usage
-----

The test group needs the ISMIP7 AIS datasets, a MALI mesh with its graph
partition file, and a MALI build that supports the ISMIP7 melt method.
Supply them in a user config file:

.. code-block:: cfg

    [paths]
    # only needed if the compass MALI-Dev submodule does not yet have the
    # ISMIP7 melt method
    mpas_model = /path/to/E3SM/components/mpas-albany-landice

    [ismip7_calibration]
    base_path_ismip7 = /path/to/ISMIP7/data/AIS
    base_path_mali = /path/to/inputdata/glc/mpasli/mpas.ais4to20km
    mali_mesh_file = ais_4to20km.20250625.nc
    mali_mesh_name = ais_4to20km
    region_mask_file = ais_4to20km_region_mask.20230105.nc
    graph_file_prefix = mpasli.graph.info.240507.part.

Then set up and run, for example:

.. code-block:: bash

    compass setup -t landice/ismip7_calibration/replication \
        -f my.cfg -w $WORKDIR
    cd $WORKDIR/landice/ismip7_calibration/replication
    sbatch job_script.sh

The ``ais`` test case creates one step per ocean state and melt form -- 56
with the default settings -- each a short MALI run.  They can be run together
with ``compass run`` from the test case directory, or individually from each
step directory.

Config options
--------------

The default config options are:

.. code-block:: cfg

    [ismip7_calibration]

    # Which ocean states to use. Options:
    #   minimal     - 7 states, those marked X in protocol Table 2
    #   recommended - 11 states, adding the regional pairs marked +
    #   all         - 28 states, everything ISMIP7 distributes
    ocean_state_subset = all

    # Which melt forms to calibrate, comma separated
    melt_forms = ismip7, ismip6

    # Number of MPI tasks for ESMF_RegridWeightGen
    esmf_ntasks = 128

    # Number of MPI tasks for each MALI run
    ntasks = 128

    # Length of the single timestep the melt diagnostic takes
    timestep = 0000-00-01_00:00:00

    [ismip7_calibration_melt]

    # Practical salinity at the ice draft, PSU
    salinity = 34.5

    # The sin(theta) factor of the ISMIP7 quadratic
    sin_slope = 0.0051117

    # Magnitude of the Coriolis parameter, s^-1
    coriolis = 1.4e-4

    [ismip7_calibration_objective]

    # Number of random draws of the term weights and the targets
    sample_size = 100000

    # Seed for the random draws, so the percentiles are reproducible
    seed = 0

    # Which summands carry non-zero weight: 'published' or 'all'
    t3_models = all
    t4_regions = all
    t4_years = all

See ``compass/landice/tests/ismip7_calibration/ismip7_calibration.cfg`` for
the full set with comments.

The ``t4_regions`` option is worth understanding.  The published weighting
uses Pine Island alone, which is 2 of the 18 available observations.  So
weighted, J4 constrains a single amplitude that either melt form can match by
rescaling its parameter, so it cannot discriminate between them.  Including
Dotson makes J4 a relative constraint between two shelves, which no rescaling
can satisfy if the ratio is wrong.  The default is therefore ``all``; set
``published`` to reproduce the published weighting.

Similarly, the reduced ocean-state subsets leave J4 structurally
under-powered, which is why ``ocean_state_subset`` defaults to ``all``.
