.. _dev_ocean_planar_convergence:

planar_convergence
==================

The ``planar_convergence`` test group
(:py:class:`compass.ocean.tests.planar_convergence.PlanarConvergence`)
implements convergence studies on the full globe. Currently, the only test case
is the advection of a cosine bell.

.. _dev_ocean_planar_convergence_horizontal_advection:

horizontal_advection
--------------------

The :py:class:`compass.ocean.tests.planar_convergence.horizontal_advection.HorizontalAdvection`
test performs a series of 24-day runs that advect a Gaussian-shaped tracer blob
across the periodic boundaries and back to its original location.  The
resolution of the sphere varies (by default, between 2 and 32 km).  Advected
results are compared with a known exact solution to determine the rate of
convergence.  See :ref:`ocean_planar_convergence_horizontal_advection`.
for config options and more details on the test case.

init
~~~~

The class :py:class:`compass.ocean.tests.planar_convergence.horizontal_advection.init.Init`
defines a step for setting up the mesh and initial state for each test case
with a tracer distributed in a Gaussian shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.planar_convergence.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step is determined from the resolution
based on the ``dt_1km`` config option.  Other namelist options are taken
from the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.planar_convergence.horizontal_advection.analysis.Analysis`
defines a step for computing the RMSE (root-mean-squared error) for the results
at each resolution and plotting them in ``convergence.png``.
