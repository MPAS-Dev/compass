.. _landice_ensemble_generator:

ensemble_generator
==================

The ``landice/ensemble_generator`` test group creates ensemble of MALI
simulations with different parameter values.  The ensemble framework
sets up a user-defined number of simulations with parameter values selected
from a space-filling Sobol sequence.

A test case in this test group consists of a number of ensemble members,
and one ensemble manager.
Each ensemble member is a step of the test case, and can be run separately
or as part of the complete ensemble.  Ensemble members are identified by a
three digit run number, starting with 000.
A config file specifies the the run numbers to set up, as well as some common
information about the run configuration.

The test case can be generated multiple times to set up and run additional
runs with a different range of run numbers after being run initially. This
allows one to perform a small ensemble (e.g. 2-10 runs) to make sure results
look as expected before spending time on a larger ensemble. This also allows
one to add more ensemble members from the Sobol sequence later if UQ analysis
indicates the original sample size was insufficient.

Individual test cases will define which parameter are being sampled and
over what range.  Currently three parameters are supported:

* basal friction power law exponent

* von Mises threshold stress for calving

* calving rate speed limit

Additional parameters can be easily added in the future.
The test group currently includes a file of unit parameter values for two
parameters with 100 samples using a Sobol sequence.  The parameter
dimensionality or sample size can be increased by modifying this file and
its usage.  It also would be possible to modify the sampling strategy to
perform uniform parameter sensitivity tests.

``compass setup`` will set up the set up simulations and the ensemble manager.
``compass run`` from the test case work directory will submit each run as a
separate slurm job.
Individual runs can be run independently through ``compass run`` executed in the
run directory.  (E.g., if you want to test or debug a run without running the
entire ensemble.)

.. note::

   Due to the requirement that ``compass run`` is only executed
   on a compute node, this operation has to be submitted via a batch script or
   interactive job, or compass framework code can be modified by an expert user
   to lift this restriction. (This may be addressed in the future.) 

Simulation output can be analyzed with the ``plot_ensemble.py`` visualization
script, which generates plots of basic quantities of interest as a function
of parameter values, as well as identifies runs that did not reach the
target year.

Future improvements may include:

* enabling the ensemble manager to identify runs that need to be restarted
  so the restarts do not need to be managed manually

* safety checks or warnings before submitting ensembles that will use large
  amounts of computing resources

* more flexibility in customizing ensembles without needing to modify test
  case files

The test group includes a single test case that creates an ensemble of Thwaites
Glacier simulations.

config options
--------------
Test cases in this test group have the following common config options.

A config file specifies the location of the input file, the basal friction
exponent used in the input file, the name of the parameter vector file to
use, and the start and end run numbers to set up.
This test group is intended for expert users, and it is expected that it
will typically be run with a customized cfg file.  Note the default run
numbers create a small ensemble, but uncertainty quantification applications
will typically need dozens or more simulations.

.. code-block:: cfg

   # config options for setting up an ensemble
   [ensemble]

   # start and end numbers for runs to set up and run
   # Additional runs can be added and run to an existing ensemble
   # without affecting existing runs, but trying to set up a run
   # that already exists may result in unexpected behavior.
   # Run numbers should be zero-based
   # These values do not affect viz/analysis, which will include any
   # runs it finds.
   start_run = 0
   end_run = 3

   param_vector_filename = Sobol_Initializations_seed_4_samples_100.csv

   input_file_path = /global/cfs/cdirs/fanssie/MALI_projects/Thwaites_UQ/Thwaites_4to20km_r02_20230126/Thwaites_4to20km_r02_20230126.nc
   # Base path to the input files. User has to supply.  This path needs to include the files listed below.  Eventually this could be hard-coded to use files on the input data server, but initially we want flexibility to experiment with different inputs and forcings

   # the value of the friction exponent used for the calculation of muFriction
   # in the input file
   orig_fric_exp = 0.2

thwaites
--------

``landice/ensemble_generator/thwaites`` uses the ensemble framework to create
and ensemble of 4 km resolution Thwaites Glacier simulations integrated from
2000 to 2100 with two parameters varying:

* basal friction power law exponent: range [0.1, 0.333]

* von Mises threshold stress for calving: range [150, 350] kPa

The initial condition file is specified in the ``ensemble_generator.cfg`` file
or a user modification of it.  The forcing files for the simulation are
hard-coded in the test case streams file  and are located on the NERSC
filesystem.  
The model configuration uses:

* first-order velocity solver

* power law basal friction

* evolving temperature

* von Mises calving

* ISMIP6 surface mass balance and sub-ice-shelf melting using climatological
  mean forcing
