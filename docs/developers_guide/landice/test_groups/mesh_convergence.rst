.. _dev_landice_mesh_convergence:

mesh_convergence
================

The ``mesh_convergence`` test group
(:py:class:`compass.landice.tests.mesh_convergence.MeshConvergence`)
implements a suite of spatial convergence tests for MALI on planar, doubly
periodic hexagonal meshes (see :ref:`landice_mesh_convergence`).  Here we
describe the shared framework and the individual test cases.

.. _dev_landice_mesh_convergence_framework:

framework
---------

The test group shares several base classes and config/namelist/streams files
that are used by all three test cases.

ConvTestCase
~~~~~~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.conv_test_case.ConvTestCase`
is the shared base class for all convergence test cases.  It reads the list of
resolutions from ``[mesh_convergence] resolutions`` and calls
:py:meth:`~compass.landice.tests.mesh_convergence.conv_test_case.ConvTestCase._setup_steps`
to construct the appropriate ``init`` and ``forward`` steps for each
resolution.  Each child class must implement ``create_init`` and
``create_analysis`` to supply the test-case-specific init and analysis step
objects.

The ``configure`` method re-runs ``_setup_steps`` in case the user has changed
the resolution list in a config file, and then calls ``update_cores`` to set
the number of MPI tasks for each forward step based on the mesh size and the
``goal_cells_per_core`` / ``max_cells_per_core`` config options.

ConvInit
~~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.conv_init.ConvInit`
is the shared base class for all init steps.  Its ``run`` method creates a
planar hexagonal mesh of the requested resolution using
``mpas_tools.planar_hex.make_planar_hex_mesh``, culls and converts the mesh,
centers it, and writes ``mesh.nc`` and ``graph.info``.  Each test case
provides a child class of ``ConvInit`` that calls ``super().run()`` and then
adds the appropriate initial conditions for that test.

ConvAnalysis
~~~~~~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.conv_analysis.ConvAnalysis`
is the shared base class for all analysis steps.  Its ``__init__`` method
registers the forward-step output files as analysis inputs using the naming
convention ``{resolution}km_output.nc → ../{resolution}km/forward/output.nc``.
Each test case provides a child class that implements the ``run`` method and,
typically, an ``rmse`` helper method.

Forward
~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.forward.Forward`
is shared by all test cases.  It reads the common ``namelist.forward`` and
``streams.forward`` files from the ``mesh_convergence`` package, and
additionally reads test-case-specific namelist/streams files if they exist in
the child test-case package.  The time step for each resolution is set via the
``get_dt_duration`` method to satisfy an advective CFL condition based on
``target_velocity`` and scaled proportionally to each mesh resolution.

Config / namelist / streams files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mesh_convergence.cfg`` sets default values for options shared across all
test cases (resolutions, domain size, core counts, etc.).  Each test case
additionally ships its own ``<name>.cfg`` with test-specific defaults.

``namelist.forward`` sets common MALI namelist options (e.g. third-order
spatial and temporal advection schemes) and is applied first; test-case
namelist files are applied on top to allow overrides.

``streams.forward`` and ``streams.template`` define the ``input``
and ``output`` streams; the output interval is filled in by the ``Forward``
step from the computed run duration.

.. _dev_landice_mesh_convergence_test_cases:

test cases
----------

horizontal_advection
~~~~~~~~~~~~~~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection.HorizontalAdvection`
tests convergence of passive-tracer advection.

Init
(:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection.init.Init`)
sets up a uniform ice slab with a Gaussian blob of ``passiveTracer2d`` at the
domain center plus a spatially uniform prescribed velocity
(``uReconstructX``, ``uReconstructY``) chosen so that the tracer completes
exactly one circuit of the periodic domain in the configured run duration.

Analysis
(:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection.analysis.Analysis`)
computes the RMSE of ``passiveTracer2d`` between the initial and final states
at each resolution, fits a power law, and saves ``convergence.png``.

horizontal_advection_thickness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection_thickness.HorizontalAdvectionThickness`
is structurally identical to ``horizontal_advection`` but tests convergence of
ice-thickness advection instead.

Init
(:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection_thickness.init.Init`)
places a Gaussian bump of ice thickness on top of a uniform background layer,
with the same prescribed velocity approach as the tracer test.

Analysis
(:py:class:`compass.landice.tests.mesh_convergence.horizontal_advection_thickness.analysis.Analysis`)
computes the RMSE of ``thickness`` between the initial and final states.

halfar
~~~~~~

:py:class:`compass.landice.tests.mesh_convergence.halfar.Halfar`
tests convergence of the ice-dynamics solver by comparing against the Halfar
analytic similarity solution.

Init
(:py:class:`compass.landice.tests.mesh_convergence.halfar.init.Init`)
calls
:py:func:`compass.landice.tests.dome.setup_mesh.setup_dome_initial_conditions`
to create the Halfar dome initial condition on the planar hexagonal mesh.

Analysis
(:py:class:`compass.landice.tests.mesh_convergence.halfar.analysis.Analysis`)
computes two error metrics:

* The RMSE between simulated and analytic Halfar thickness over all cells
  that contain ice in either solution (``convergence_rmse.png``).
* The absolute error at the dome center (``convergence_dome.png``).

The order of convergence is derived from the RMSE metric and compared against
``conv_thresh`` (fail below) and ``conv_max`` (warn above) config options.
