.. _dev_ocean_global_convergence:

global_convergence
==================

The ``global_convergence`` test group
(:py:class:`compass.ocean.tests.global_convergence.GlobalConvergence`)
implements convergence studies on the full globe. Currently, the only test case
is the advection of a cosine bell.

.. _dev_ocean_global_convergence_mesh_types:

mesh types
----------

The global convergence test cases support two types of meshes: ``qu`` meshes
created with the :py:class:`compass.mesh.QuasiUniformSphericalMeshStep` step
and ``icos`` meshes created with
:py:class:`compass.mesh.IcosahedralMeshStep`.  In general, the ``icos`` meshes
are more uniform but the ``qu`` meshes are more flexible.  The ``icos`` meshes
only support a fixed set of resolutions described in
:ref:`dev_spherical_meshes`.

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

This step builds a global mesh with uniform resolution. The type of mesh
depends on the mesh type (``qu`` or ``icos``).

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
