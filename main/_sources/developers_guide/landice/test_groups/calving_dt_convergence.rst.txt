.. _dev_landice_calving_dt_convergence:

calving_dt_convergence
======================

The ``calving_dt_convergence`` test group
(:py:class:`compass.landice.tests.calving_dt_convergence.CalvingDtConvergence`)
implements tests for assessing timestep convergence of calving physics
(see :ref:`landice_calving_dt_convergence`). Here,
we describe the shared framework for this test group.

.. _dev_landice_calving_dt_convergence_framework:

framework
---------

Additionally, the test group has a shared ``namelist.landice`` file with
appropriate namelist options, and a ``streams.landice.template`` file that
defines ``input``, ``restart``, ``output``, and ``globalStats`` streams.
The ``input`` stream uses a template to fill in the appropriate input file
for each mesh available.

DtConvergenceTest
~~~~~~~~~~~~~~~~~

The class
:py:class:`compass.landice.tests.calving_dt_convergence.DtConvergenceTest`
defines a procedure for setting up a series of runs with different values for
``config_adaptive_timestep_calvingCFL_fraction``.
The list of fractions to use is hardcoded, with a smaller number of values
when using the FO velocity solver, because of their much larger cost.

The ``validate`` method analyzes the resulting runs and plots the summary.
Note that because the analysis is part of the ``validate`` method, it will
only occur if all runs have completed successfully, and it cannot be run
manually.  This can be annoyance for some of the FO velocity tests that take a
long time to run if an error occurs.  Given the occasional usage of this test
case, that is fine, but if it becomes a problem, the analysis could be moved
to a separate step.

RunModel
~~~~~~~~

The class
:py:class:`compass.landice.tests.calving_dt_convergence.run_model.RunModel`
defines a step for running MALI for each value of
``config_adaptive_timestep_calvingCFL_fraction`` specified.  It includes logic
for handling the different meshes, calving, and velocity options in the
namelist and streams files.
