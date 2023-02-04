.. _dev_landice_ensemble_generator:

ensemble_generator
==================

The ``ensemble_generator`` test group (:py:class:`compass.landice.tests.ensemble_generator`)
creates an ensemble of MALI
simulations with different parameter values.  The ensemble framework
sets up a user-defined number of simulations with parameter values selected
from a space-filling Sobol sequence
(see :ref:`landice_ensemble_generator`).

.. _dev_landice_ensemble_generator_framework:

framework
---------

The shared config options for the ``ensemble_generator`` test group are described
in :ref:`landice_ensemble_generator` in the User's Guide.
See the ``thwaites`` test case description below for how this framework should
be used to set up an ensemble.

ensemble_member
~~~~~~~~~~~~~~~
The class :py:class:`compass.landice.tests.ensemble_generator.EnsembleMember`
defines a step for a single ensemble member (model run).  The constructor
stores the run number and name, as well as the parameter values to be used
and the python package where the namelist, streams, and albany input file
can be found.  By setting up an ``ensemble_member`` this way, this class can
be flexibly called from any test case that one would like to define.

The ``setup`` method actually sets up the run by first setting up a baseline
configuration and then modifying parameter values using the parameter
values defined when the constructor was called.  There are steps to set
parameters:

* basal friction exponent

* von Mises threshold stress

* calving speed limit

These parameter values are always set; if it desired they not be modified
from the baseline configuration, a default value should be passed to the
constructor.

Because changing the exponent requires modifying the input file to adjust
muFriction to yield the same basal shear stress as the original file,
there is also an operation to set the output filename in the streams file,
where the file to be used and modified was specified in the cfg file for the
test case.

Additionally, a job script is written for the run so that the run can be
submitted as a slurm job independent of other runs in the ensemble.  This also
allows a user to easily run a single ensemble member by submitting the job
script within the run step work directory.

Finally, a symlink to the compass load script is added to the run work
directory, which compass does not do by default.

The ``run`` method creates a graph file and runs MALI.

Finally, there is a function ``_adjust_friction_exponent`` that modifies the
friction exponent in the ``albany_input.yaml`` file and adjusts muFriction
in the input file to maintain an unchanged basal shear stress.

ensemble_manager
~~~~~~~~~~~~~~~~
The class :py:class:`compass.landice.tests.ensemble_generator.EnsembleManager`
defines a step for managing the entire ensemble.  The constructor and setup
methods perform minimal operations.  The ``run`` method submits each run in
the ensemble as a slurm job.  Eventually the ensemble_manager will be able
to assess if runs need restarts and modify them to be submitted as such.

thwaites
--------

The :py:class:`compass.landice.tests.ensemble_generator.thwaites.ThwaitesEnsemble`
uses the framework described above to set up an ensemble of Thwaites Glacier
simulations.  The constructor simply adds the ensemble manager as the only step.
This allows the test case to be listed by ``compass list`` without having all
ensemble members listed in a verbose listing.  Because there may be dozens of
ensemble members, it is better to wait to have them added until the setup
phase.  Also, by waiting until configure to define the ensemble members, it
is possible to have the start and end run numbers set in the config,
because the config is not parsed by the constructor.

The ``configure`` method is where most of the work happens.  Here, the start and
end run numbers are read from the config, the parameter array is read from
file, and the parameters to be varied and over what range is defined.
Basal friction exponent and von Mises threshold stress are varied over set
ranges.  The calving speed limit is set to a fixed value for all runs,
but could be included as a varying parameter in the future. 
The values for each parameter are
passed to the ``EnsembleMember`` constructor to define each run.
Finally, each run is now added to the test case as a step to run,
because they were not automatically added by compass during the test
case constructor phase.

Note, the ``configure`` method could be modified to use other parameter
value sampling techniques than the provided 2d Sobol sequence.  For
example, a sensitivity test could be performed by defining one parameter's
values using the numpy arange function or similar.

There are no ``run`` or ``validate`` steps required.  The ensemble manager
handles "running" ensemble members by submitting them as slurm jobs.
This is a major difference in how this test case functions from most
compass test cases.
The visualization script ``plot_ensemble.py`` is symlinked in the test
case work directory and can be run manually to assess the status of the
ensemble, but there is not a formal analysis step that can be run through
compass.
