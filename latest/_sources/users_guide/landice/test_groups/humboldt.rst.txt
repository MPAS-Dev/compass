.. _landice_humboldt:

humboldt
========

The ``landice/humboldt`` test group includes  test case for creating a
mesh for Humboldt Glacier, Greenland, and a series of tests for running
simulations on a Humboldt Glacier mesh.  Note that the tests that run MALI do
not use the output of the mesh generation step directly, but instead
download a pre-generated initial condition file.  This is because some of the
initialization steps still happen outside of COMPASS, most notably the
optimization for basal friction, which requires expert usage and takes a
larger amount of computing resources than COMPASS is typically run with.

The tests that run MALI are set up in a way that creates a separate version
of each test with each calving law that MALI currently supports.  There are
versions of each with either the FO velocity solver or no velocity solver,
in which case a modeled velocity field from the initial condition is used.
There are also two meshes available for running with MALI: a fast 3 km
resolution mesh and a slower but more accurate 1 km mesh.  Most tests use the
faster 3 km mesh.
There also is a test that runs the subglacial hydrology model without any
other physics.

.. figure:: images/humboldt_1to10km.png
   :width: 777 px
   :align: center

   Ice thickness on Humboldt 1-10km variable resolution mesh.

config options
--------------

The test group uses the following default config options.  At this point only
the mesh generation options are adjusted through the config file.

.. code-block:: cfg

    [mesh]

    # number of levels in the mesh
    levels = 10

    # distance from ice margin to cull (km).
    # Set to a value <= 0 if you do not want
    # to cull based on distance from margin.
    cull_distance = 5.0

    # mesh density parameters
    # minimum cell spacing (meters)
    min_spac = 1.e3
    # maximum cell spacing (meters)
    max_spac = 1.e4
    # log10 of max speed (m/yr) for cell spacing
    high_log_speed = 2.5
    # log10 of min speed (m/yr) for cell spacing
    low_log_speed = 0.75
    # distance at which cell spacing = max_spac (meters)
    high_dist = 1.e5
    # distance within which cell spacing = min_spac (meters)
    low_dist = 1.e4

    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = False
    use_dist_to_edge = True

mesh_gen
--------

``landice/humboldt/default`` creates a 1-10km variable resolution mesh. 
There is no model integration step.

decomposition_tests
-------------------

There are a number of variants of a decomposition test that runs a 5-year
simulation on 16 (16proc_run step) and then on 32 cores (32proc_run step)
to make sure key prognostic variables are either bit-fot-bit (without the
FO solver) or have only small differences within a specified tolerance (with
the FO solver).  The FO solver is not BFB on different decompositions, but the
differences are small.  There are variants of this test for each calving law
that MALI currently supports, paired with either the FO velocity solver or no
velocity solver.
The full set of combinations use the 3 km mesh.  There is additionally a
decomposition test using the 1 km mesh that has calving disabled.
Finally, there is a set of "full physics" tests that use von Mises calving,
plus damage threshold calving and marine facemelting.  This configuration can
be run with either the FO velocity solver or no velocity solver.  It is meant
to exercise the widest range of physics currently supported in MALI.  To make
this test faster for the integration test suite, it uses a 6-month time step
instead of 4 months.

restart_tests
-------------

There are a number of variants of a restart test that runs a 3-year simulation
compared to a 2-year simulation followed by a restart for an additional
1 year.  Results should be bit-for-bit identical.  
There are variants of this test for each calving law
that MALI currently supports, paired with either the FO velocity solver or no
velocity solver.
The full set of combinations use the 3 km mesh.  There is additionally a
decomposition test using the 1 km mesh that has calving disabled.
Finally, there is a set of "full physics" tests that use von Mises calving,
plus damage threshold calving and marine facemelting.  This configuration can
be run with either the FO velocity solver or no velocity solver.  It is meant
to exercise the widest range of physics currently supported in MALI.  To make
this test faster for the integration test suite, it uses a 6-month time step
instead of 4 months.
