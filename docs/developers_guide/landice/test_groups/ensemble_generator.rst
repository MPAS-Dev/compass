.. _dev_landice_ensemble_generator:

ensemble_generator
==================

The ``ensemble_generator`` test group (:py:class:`compass.landice.tests.ensemble_generator`)
creates an ensemble of MALI
simulations with different parameter values.  The ensemble framework
sets up a user-defined number of simulations with parameter values selected
by uniform sampling or from a space-filling Sobol sequence
(see :ref:`landice_ensemble_generator`).

.. _dev_landice_ensemble_generator_framework:

framework
---------

The shared config options for the ``ensemble_generator`` test group are described
in :ref:`landice_ensemble_generator` in the User's Guide.

ensemble_member
~~~~~~~~~~~~~~~
The class :py:class:`compass.landice.tests.ensemble_generator.EnsembleMember`
defines a step for a single ensemble member (model run).  The constructor
stores the run number and name, as well as the parameter values to be used
and the python package where the namelist, streams, and albany input file
can be found.  By setting up an ``ensemble_member`` this way, this class can
be flexibly called from any ensemble test case variants that one would like
to define in the future.

The ``setup`` method actually sets up the run by first setting up a baseline
configuration and then modifying parameter values using the parameter
values defined when the constructor was called.  There are operations to set
parameters:

* basal friction exponent

* scaling factor on muFriction

* scaling factor on stiffnessFactor

* von Mises threshold stress

* calving speed limit

* gamma0 melt sensitivity parameter in ISMIP6-AIS ice-shelf basal melting
  parameterization

* target ice-shelf basal melt rate for ISMIP6-AIS ice-shelf basal melting
  parameterization.  In the model setup, the deltaT thermal forcing bias
  adjustment is adjusted to obtain the target melt rate for a given gamma0

Each parameter can be activated or disabled as a free parameter.  If disabled,
whatever values specified in namelist and input files will be used.

Because changing the exponent requires modifying the input file to adjust
muFriction to yield the same basal shear stress as the original file,
there is also an operation to set the output filename in the streams file,
where the file to be used and modified was specified in the cfg file for the
test case.  The default input file is renamed to indicate it was modified.
This is updated automatically in the streams
file, and it seeks to avoid potential confusion if a user were to use this
file for another purpose.

Similarly, because changing gamma0 and deltaT require modifying a basal melt
parameter file, the baseline file path needs to be specified in the config
and that file is copied, renamed, and modified in each run directory.

Additionally, a job script is written for the run so that the run can be
submitted as a slurm job independent of other runs in the ensemble.  This also
allows a user to easily run a single ensemble member by submitting the job
script within the run step work directory.

Finally, a symlink to the compass load script is added to the run work
directory, which compass does not do by default.

The ``run`` method creates a graph file and runs MALI.

There is a function ``_adjust_friction_exponent`` that modifies the
friction exponent in the ``albany_input.yaml`` file and adjusts muFriction
in the input file to maintain an unchanged basal shear stress.  Similarly,
there is a function ``_adjust_basal_melt_params`` that modifes gamma0 and
deltaT in a basal melt parameter file.

ensemble_manager
~~~~~~~~~~~~~~~~
The class :py:class:`compass.landice.tests.ensemble_generator.EnsembleManager`
defines a step for managing the entire ensemble.  The constructor and setup
methods perform minimal operations.  The ``run`` method submits each run in
the ensemble as a slurm job.  Eventually the ``ensemble_manager`` will be able
to assess if runs need restarts and modify them to be submitted as such.

spinup_ensemble
---------------

The :py:class:`compass.landice.tests.ensemble_generator.spinup_ensemble.SpinupEnsemble`
uses the framework described above to set up an ensemble of spinup or historical
simulations from a common initial condition.
The constructor simply adds the ensemble manager as the only step.
This allows the test case to be listed by ``compass list`` without having all
ensemble members listed in a verbose listing.  Because there may be dozens of
ensemble members, it is better to wait to have them added until the setup
phase.  Also, by waiting until configure to define the ensemble members, it
is possible to have the start and end run numbers set in the config,
because the config is not parsed by the constructor.

The ``configure`` method is where most of the work happens.  Here, the start
and end run numbers are read from the config, a parameter array is generated,
and the parameters to be varied and over what range are defined.
The values for each parameter are
passed to the ``EnsembleMember`` constructor to define each run.
Finally, each run is now added to the test case as a step to run,
because they were not automatically added by compass during the test
case constructor phase.

The ``run`` step simply sets up a graph file and runs the model.

The ensemble manager
handles "running" ensemble members by submitting them as slurm jobs.
This is a major difference in how this test case functions from most
compass test cases.
The visualization script ``plot_ensemble.py`` is symlinked in the test
case work directory and can be run manually to assess the status of the
ensemble, but there is not a formal analysis step that can be run through
compass.

branch_ensemble
---------------

The :py:class:`compass.landice.tests.ensemble_generator.branch_ensemble.BranchEnsemble`
sets up an ensemble of runs each of which are branched from an ensemble
member of a previously run spinup ensemble.
The constructor adds the ensemble_manager as a step, as with the spinup_ensemble.

The ``configure`` method searches over the range of runs requested and assesses if
the corresponding spinup_ensemble member reached the requested branch time.
If so, and if the branch_ensemble memebr directory does not already exist, that
run is added as a step.  Within each run (step), the restart file from the branch
year is copied to the branch run directory.  The time stamp is reassigned to
2015 (this could be made a cfg option in the future).  Also copied over are
the namelist and albany_input.yamlm files.  The namelist is updated with
settings specific to the branch ensemble, and a streams file specific to the
branch run is added.  Finally, details for managing runs are set up, including
a job script.

As in the spinup_ensemble, the ``run`` step just runs the model.
