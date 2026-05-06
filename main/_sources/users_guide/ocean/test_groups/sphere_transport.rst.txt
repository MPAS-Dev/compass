.. _ocean_sphere_transport:

sphere_transport
==================

The ``sphere_transport`` test group implements 4 test cases for standard
transport schemes.   The test cases are convergence studies on the full globe,
similar to :ref:`ocean_global_convergence`. Most tests are from
`Lauritzen et al. (2012) <https://gmd.copernicus.org/articles/5/887/2012/>`_.

All test-case results include solution plots for each resolution and
convergence plots. Convergence data and mass conservation error is written to
``analysis.log`` in csv format for each test case.

The test cases use MPAS's ``debugTracers`` to define 3 tracers.   In all cases
except CorrelatedTracers2D (discussed below), the 3 tracers are:

- Tracer1: A c-infinity function used for convergence analysis
- Tracer2: A pair of c-2 cosine bells
- Tracer3: A discontinuous pair of slotted cylinders

.. image:: images/sphere_transport.png
   :width: 500 px
   :align: center

.. _ocean_sphere_transport_rotation_2d:

rotation_2d
-----------

The ``rotation_2d`` test case has the simplest velocity field: rigid rotation
about an axis offset from the z-axis of the sphere.  It is the only test case
not from Lauritzen et al. (2012).

config options
~~~~~~~~~~~~~~

Here, we give config options for the ``rotation_2d`` test case.  Config options
for the other test cases are nearly identical:

.. code-block:: cfg

    # options for rotation2D convergence test case
    [rotation_2d]

    # a list of resolutions (km) to test
    resolutions = 60, 90, 120, 150, 180, 240

    # time step in minutes (1-1 with resolutions)
    timestep_minutes = 8, 12, 16, 20, 24, 32

    # the number of cells per core to aim for
    goal_cells_per_core = 300

    # the approximate maximum number of cells per core (the test will fail if too
    # few cores are available)
    max_cells_per_core = 3000

    # time integrator (RK4 or split_explicit)
    time_integrator = RK4

    # convergence threshold below which the test fails
    tracer1_conv_thresh = 1.4
    tracer2_conv_thresh = 1.8
    tracer3_conv_thresh = 0.4

resolutions
~~~~~~~~~~~

To alter the resolutions used in this test, you will need to create your own
config file (or add a ``rotation_2d`` section to a config file if you're
already using one).  The resolutions are a comma-separated list of the
quasi-uniform resolution of the mesh in km.  If you specify a different list
before setting up ``rotation_2d``, steps will be generated with the requested
resolutions.  (If you alter ``resolutions`` in the test case's config file in
the work directory, nothing will happen.)

time step
~~~~~~~~~

For each resolution, the time step in minutes is provided in
``timestep_minutes``.  You can alter this before setup (in a user config file)
or before running the test case (in the config file in the work directory).

The ``time_integrator`` config option determines whether 4th-order Runge-Kutta
(`RK4`) or split-explicit (`split_explicit`) time integration is performed.

cores
~~~~~

The number of cores (and the minimum) is proportional to the number of cells,
so that the number of cells per core is roughly constant.  You can alter how
many cells are allocated to each core with ``goal_cells_per_core``.  You can
control the maximum number of cells that are allowed to be placed on a single
core (before the test case will fail) with ``max_cells_per_core``.  If there
aren't enough processors to handle the finest resolution, you will see that
the step (and therefore the test case) has failed.

convergence thresholds
~~~~~~~~~~~~~~~~~~~~~~

The three convergence thresholds are used to determine if the error in each of
the tracers converges at the expected rate.  If a linear fit to the error as
a function of resolution has a slope that is shallower than the specified
value, the test case will fail, raising an error.  The idea is to detect
changes in the code that appreciably degrade performance of tracer transport.
If you change the resolutions used in the test case, you may need to adjust
these thresholds.

.. _ocean_sphere_transport_nondivergent_2d:

nondivergent_2d
---------------

Results from the ``nondivergent_2d`` test case also include the "filament norm"
and over-/under-shoot analysis.  The reversing deformational flow in this
test case explores large-scale to small-scale transport.

.. _ocean_sphere_transport_divergent_2d:

divergent_2d
------------

The ``divergent_2d`` test case analyzes transport scheme with a velocity field
that has nonzero divergence.

.. _ocean_sphere_transport_correlated_tracers_2d:

correlated_tracers_2d
---------------------

The ``correlated_tracers_2d`` test case uses the velocity as
:ref:`ocean_sphere_transport_nondivergent_2d`, but instead of slotted cylinders
in the 3rd tracer, adds a second pair of cosine bells that are nonlinearly
correlated with tracer2.  This tests the ability of the transport schemes to
maintain such a correlation.


