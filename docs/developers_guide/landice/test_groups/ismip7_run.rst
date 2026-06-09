.. _dev_landice_ismip7_run:

ismip7_run
==========

The ``ismip7_run`` test group
(:py:class:`compass.landice.tests.ismip7_run`) sets up experiments from
the ISMIP7 experimental protocol for both the Antarctic Ice Sheet (AIS)
and the Greenland Ice Sheet (GrIS).  Optionally, the AIS test case
supports coupled MALI–Sea Level Model (SLM) simulations.
(see :ref:`landice_ismip7_run`).

framework
---------

The ``ismip7_run`` test group
(:py:class:`compass.landice.tests.ismip7_run.Ismip7Run`) registers two
test cases:

* :py:class:`compass.landice.tests.ismip7_run.ismip7_ais.Ismip7Ais`
* :py:class:`compass.landice.tests.ismip7_run.ismip7_gris.Ismip7Gris`

There is no shared functionality between the two test cases at present.
Shared functions may be added in the future if the needed functionality
can be generalized.

ismip7_ais
----------

The :py:class:`compass.landice.tests.ismip7_run.ismip7_ais.Ismip7Ais`
test case sets up an ensemble of ISMIP7 Antarctica simulations
(standalone MALI or coupled MALI-SLM).

The constructor (``__init__``) does nothing other than allow the
``ismip7_ais`` test case to be listed by ``compass list`` without having
all individual experiments listed in a verbose listing.  Each individual
experiment is a step rather than a test case to avoid excessive
subdirectories.

The ``configure`` method parses the ``exp_list`` config option from the
``[ismip7_run_ais]`` section. It supports:

* ``all`` — all 11 core experiments
* ``historical`` — just the two historical runs
* ``projections`` — the six SSP projection runs
* ``ctrl`` — the two CTRL2015 runs
* A comma-delimited list of specific experiment names

Each selected experiment is added as a
:py:class:`~compass.landice.tests.ismip7_run.ismip7_ais.set_up_experiment.SetUpExperiment`
step and immediately removed from ``steps_to_run`` (experiments should be
submitted individually, not run through the test case).

The ``run`` method raises an error instructing the user to submit batch
jobs for each experiment individually.

set_up_experiment (AIS)
~~~~~~~~~~~~~~~~~~~~~~~

The class
:py:class:`compass.landice.tests.ismip7_run.ismip7_ais.set_up_experiment.SetUpExperiment`
defines a step for a single ISMIP7 AIS experiment.

The ``setup`` method sets up the experiment directory by:

1. Creating symlinks to forcing files from the conventional path layout
   under ``forcing_basepath``.
2. Copying and populating the streams template with the correct forcing
   filenames and intervals (monthly for SMB/temperature/runoff, annual
   for lapse rates and thermal forcing, ``initial_only`` for melt
   parameters).
3. Processing the namelist template for the experiment's time period
   and restart frequency.
4. Adding calving-specific streams (face melting, von Mises params) if
   configured.
5. Creating a restart symlink for projection experiments pointing to
   the corresponding ESM's historical restart
   (``../historical_{model}/rst.2015-01-01.nc``).
6. Setting up CTRL2015 experiments with constant-climate forcing
   (``initial_only`` intervals).
7. Setting up the OCX experiment with reanalysis-based forcing.
8. If SLM coupling is enabled, adding a ``CreateSlmMappingFiles`` step
   and writing the SLM namelist from the Jinja2 template.
9. Generating a ``graph.info`` file and a SLURM job script.
10. Symlinking the compass load script into the run directory.

The ``run`` method executes MALI for the given experiment.

create_slm_mapping_files
~~~~~~~~~~~~~~~~~~~~~~~~

The class
:py:class:`compass.landice.tests.ismip7_run.ismip7_ais.create_slm_mapping_files.CreateSlmMappingFiles`
creates mapping files between the MALI mesh and the SLM grid.  This step
is only added when sea-level model coupling is enabled.

ismip7_gris
-----------

The :py:class:`compass.landice.tests.ismip7_run.ismip7_gris.Ismip7Gris`
test case mirrors ``ismip7_ais`` but for the Greenland Ice Sheet.

Key differences from the AIS test case:

* Ocean thermal forcing is 2D (depth-averaged) rather than 3D.
* No sea-level model coupling.
* Default calving method is ``von_mises``.
* Config section is ``[ismip7_run_gris]``.

set_up_experiment (GrIS)
~~~~~~~~~~~~~~~~~~~~~~~~

The class
:py:class:`compass.landice.tests.ismip7_run.ismip7_gris.set_up_experiment.SetUpExperiment`
follows the same logic as the AIS version, with the differences noted
above (2D TF stream, no SLM support).
