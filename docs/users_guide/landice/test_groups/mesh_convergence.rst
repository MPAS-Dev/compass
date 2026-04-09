.. _landice_mesh_convergence:

mesh_convergence
================

The ``landice/mesh_convergence`` test group includes tests for assessing the
spatial (mesh) convergence of MALI on planar, doubly periodic hexagonal
meshes.  Currently, three test cases are available:
:ref:`landice_mesh_convergence_horizontal_advection`,
:ref:`landice_mesh_convergence_horizontal_advection_thickness`, and
:ref:`landice_mesh_convergence_halfar`.

Each test case runs MALI at a series of resolutions, then computes and plots
the order of convergence of an error metric as a function of the number of
cells in the mesh.  The analysis step raises an error if the computed order of
convergence falls below a configurable threshold.

config options
--------------

All test cases in this test group share the following ``[mesh_convergence]``
config options, which may be overridden on a per-test-case basis:

.. code-block:: cfg

    # options for mesh convergence test cases
    [mesh_convergence]

    # a list of resolutions (km) to test
    resolutions = 8, 4, 2, 1

    # number of mesh cells in x and y for 1 km resolution.  Other resolutions
    # have the same physical size.
    nx_1km = 512
    ny_1km = 640

    # whether to make mesh nonperiodic or not
    nonperiodic = False

    # the number of cells per core to aim for
    goal_cells_per_core = 300

    # the approximate maximum number of cells per core
    max_cells_per_core = 3000

    # target velocity (m/yr) used to define the time step via an advective CFL
    # condition
    target_velocity = 2000.0

    # the duration (years) of the run
    duration = 1000

.. _landice_mesh_convergence_horizontal_advection:

horizontal_advection
--------------------

``landice/mesh_convergence/horizontal_advection`` tests the spatial convergence
of passive tracer advection in MALI.  A Gaussian blob of a 2-D passive tracer
(``passiveTracer2d``) is placed on a doubly periodic mesh and advected with a
prescribed, spatially uniform velocity exactly once around the domain.  The
root-mean-squared difference between the final and initial tracer fields is
then used as the error metric.

The test case config options are:

.. code-block:: cfg

    # options for planar horizontal advection test case
    [horizontal_advection]

    # Number of vertical levels
    vert_levels = 3

    # ice thickness (m)
    ice_thickness = 1000.0

    # bed elevation (m)
    bed_elevation = 0.0

    # center of the tracer gaussian (km)
    x_center = 0.
    y_center = 0.

    # width of gaussian tracer "blob" (km)
    gaussian_width = 50

    # whether to advect in x, y, or both
    advect_x = True
    advect_y = True

    # convergence threshold below which the test fails
    conv_thresh = 0.6

    # Convergence rate above which a warning is issued
    conv_max = 3.0

    # options for mesh convergence test cases
    [mesh_convergence]

    # a list of resolutions (km) to test
    resolutions = 16, 8, 4, 2

.. _landice_mesh_convergence_horizontal_advection_thickness:

horizontal_advection_thickness
-------------------------------

``landice/mesh_convergence/horizontal_advection_thickness`` tests the spatial
convergence of ice-thickness advection in MALI.  A Gaussian bump of ice
thickness (on top of a uniform background layer) is placed on a doubly
periodic mesh and advected with a prescribed, spatially uniform velocity
exactly once around the domain.  The root-mean-squared difference between the
final and initial thickness fields is used as the error metric.

The test case uses the same config options as
:ref:`landice_mesh_convergence_horizontal_advection`.

.. _landice_mesh_convergence_halfar:

halfar
------

``landice/mesh_convergence/halfar`` tests the spatial convergence of MALI's
ice dynamics by comparing the simulated ice-sheet geometry to the Halfar
analytic similarity solution :cite:`Halfar1983`.  The test begins from the
analytic Halfar ice dome (using the same initial condition as the
:ref:`landice_dome` test case with ``dome_type = halfar``) and runs the model
for a configurable duration.  Two error metrics are computed at the end of the
run:

* **RMSE** — the root-mean-squared difference between the simulated and
  analytic thickness fields over all cells that contain ice in either solution,
  plotted in ``convergence_rmse.png``.
* **dome center error** — the absolute difference between the simulated and
  analytic thickness at the dome center, plotted in ``convergence_dome.png``.

The order of convergence is estimated from the RMSE metric.  The test fails if
the order of convergence is below ``conv_thresh`` and issues a warning if it
exceeds ``conv_max``.

The test case config options are:

.. code-block:: cfg

    # options for halfar mesh convergence test
    [halfar]

    # Number of vertical levels
    vert_levels = 10

    # convergence threshold below which the test fails
    conv_thresh = 0.0

    # Convergence rate above which a warning is issued
    conv_max = 3.0

    # config options for dome test cases
    [dome]

    # the dome type ('halfar' or 'cism')
    dome_type = halfar

    # Whether to center the dome in the center of the cell that is closest to
    # the center of the domain
    put_origin_on_a_cell = True

    # whether to add a small shelf to the test
    shelf = False

    # whether to add hydrology to the initial condition
    hydro = False

    [mesh_convergence]

    # a list of resolutions (km) to test
    resolutions = 8, 4, 2, 1

    nx_1km = 128
    ny_1km = 128
    nonperiodic = True

    # target velocity (m/yr); artificially large to satisfy the more
    # restrictive diffusive CFL condition for Halfar
    target_velocity = 30000.0

    # the duration (years) of the run
    duration = 200
