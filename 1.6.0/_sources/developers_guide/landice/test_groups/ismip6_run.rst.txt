.. _dev_landice_ismip6_run:

ismip6_run
==========

The ``ismip6_run`` test group (:py:class:`compass.landice.tests.ismip6_run`)
sets up experiments from the ISMIP6 experimental protocol.
Additionally, the test group has an option to setup coupled MALI-Sea Level Model (SLM) simulations.
(see :ref:`landice_ismip6_run`).

framework
---------

There is no shared functionality for the ``ismip6_run`` test group.
Shared functions may be added if additional test cases are added and the
needed functionality can be generalized.

ismip6_ais_proj2300
-------------------

The :py:class:`compass.landice.tests.ismip6_run.ismip6_ais_proj2300.Ismip6AisProj2300`
sets up an ensemble of ISMIP6 Antarctica 2300 (standalone MALI and coupled MALI-SLM)
simulations.  The constructor (``__init__``) does nothing other than
allow the ``ismip6_ais_proj2300`` test case to be listed by ``compass list``
without having all individual experiments listed in a verbose listing.
Each individual experiment is a step rather than a test case to avoid having
excessive subdirectories.

The ``configure`` method is where the experiments to be set up are determined.
The config option ``exp_list`` is parsed and each experiment to be included
is set up as a step of the test case by calling the 
``SetUpExperiment`` constructor (``__init__``).
All of the experiments are removed from ``steps_to_run``, because the
experiments (steps) are not meant to be run together through the test
case, which is a contrary design to most compass test cases.

The ``run`` method of the test case generates an error instructing the user
to submit batch jobs for each experiment individually.

set_up_experiment
~~~~~~~~~~~~~~~~~

The class :py:class:`compass.landice.tests.ismip6_run.ismip6_ais_proj2300.set_up_experiment.SetUpExperiment`
defines a step for a single ISMIP6 experiment (model run).  The constructor
(``__init__``) stores the experiment name.

The ``setup`` method actually sets up an experiment by taking a baseline
configuration and then namelist and streams as necessary to define the ISMIP6
experiment requested.  See the ISMIP6 website for the `list of experiment
definitions <https://www.climate-cryosphere.org/wiki/index.php?title=ISMIP6-Projections2300-Antarctica#List_of_Projections>`_.

A ``graph.info`` file is created for each run.
Additionally, a job script is written for the run so that the run can be
submitted as an independent slurm job.
Finally, a symlink to the compass load script is added to the run work
directory, which compass does not do by default.

The ``run`` runs MALI for the given experiment. It also builds mapping files
between MALI and the SLM by calling a local method ``_build_mapping_files``
if the SLM option in the config file is enabled.

Important notes for analyzing coupled MALI-SLM simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
