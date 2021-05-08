.. _dev_ocean_global_convergence:

global_convergence
==================

The ``global_convergence`` test group
(:py:class:`compass.ocean.tests.global_convergence.GlobalConvergence`)
implements convergence studies on the full globe. Currently, the only test case
is the cosine bell.

.. _dev_ocean_global_convergence_cosine_bell:

cosine_bell
-----------

The :py:class:`compass.ocean.tests.global_convergence.cosine_bell.CosineBell`
test performs a series of 24-day runs that advect a bell-shaped tracer blob
around the sphere.  The resolution of the sphere varies (by default, between
60 and 240 km).  Advected results are compared with a known exact solution to
determine the rate of convergence.  See :ref:`ocean_global_convergence_cosine_bell`.
for config options and more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.global_convergence.cosine_bell.mesh.Mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.global_convergence.cosine_bell.init.Init`
defines a step for setting up the initial state for each test case with a
tracer distributed in a cosine-bell shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.global_convergence.cosine_bell.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step is determined from the resolution
based on the ``dt_per_km`` config option.  Other namelist options are taken
from the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.global_convergence.cosine_bell.analysis.Analysis`
defines a step for computing the RMSE (root-mean-squared error) for the results
at each resolution and plotting them in ``convergence.png``.
