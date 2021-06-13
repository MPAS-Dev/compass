.. _dev_ocean_gotm:

gotm
====

The ``gotm`` test group (:py:class:`compass.ocean.tests.gotm.Gotm`)
implements a test case for validating the General Ocean Turbulence Model
(`GOTM <https://gotm.net/portfolio/>`_) within MPAS-Ocean. See
:ref:`ocean_gotm` for more details.

.. _dev_ocean_gotm_default:

default
-------

The :py:class:`compass.ocean.tests.gotm.default.Default` test implements a
single column test case from Section 5.1 of
`Kärnä, 2020 <https://doi.org/10.1016/j.ocemod.2020.101619>`_.  The test runs
for 12 hours, after which the velocity and viscosity are compared against an
analytic solution.  See :ref:`ocean_gotm_default` for config options and more
details on the test case.

init
~~~~

The class :py:class:`compass.ocean.tests.gotm.default.init.Init`
defines a step for setting up the mesh and initial state for the test case. It
simply creates a small (4 x 4 cell, doubly periodic) mesh with 2.5 km
horizontal resolution.  The vertical grid divides the 15 m total ocean depth
into 250 layers of 6 cm thickness each.  Velocity, temperature and
salinity are not initialized (and are therefore zero).

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.gotm.default.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``init`` step.  The model runs for 12 hours with 25-s time steps, writing
output ever 10 minutes.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.gotm.default.analysis.Analysis`
defines a step for plotting the velocity and viscosity profiles after 12 hours,
and comparing them with an analytic solution.
