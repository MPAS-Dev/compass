.. _ocean_soma:

soma
====

The ``ocean/soma`` test group implements variants of the Simulating Ocean
Mesoscale Activity (SOMA) test case from
`Wolfram et al. (2015) <https://doi.org/10.1175/JPO-D-14-0260.1>`_.  From that
paper:

    The SOMA configuration is designed to investigate equilibrium mesoscale
    activity in a setting similar to how ocean climate models are deployed. It
    simulates an eddying, midlatitude ocean basin with latitudes ranging
    from 21.5° to 48.5°N and longitudes ranging from 16.5°W to 16.5°E. In
    contrast to previous idealized double-gyre studies...this basin is circular
    instead of rectangular and features more realistic curved coastlines with a
    150-km-wide, 100-m-deep continental shelf.

Appendix A of the paper describes the topography, initial condition and wind
forcing.

The implementation of ``soma`` in ``compass`` includes four test cases:
``default``, ``long``, ``surface_restoring``, and ``three_layer``
at 4 resolutions: ``4km``, ``8km``, ``16km`` and ``32km``, and a fifth test
case, ``particles``, that is only at ``32km`` resolution.  The test cases at all
resolutions include ``initial_state`` and ``forward`` steps, and the
``particles`` test case, which also includes particle dynamics, has an extra
``analysis`` step.  The ``initial_state`` begins with a predefined mesh,
culls out "land" cells, and defines the initial conditions for the model.
The ``forward`` step performs a short time integration of the model.  The
``analysis`` step compares the temperature and salinity at each particle with
an analytic solution defining the initial condition.

shared config options
---------------------

Both ``soma`` test cases share the following config options:

.. code-block:: cfg

    # config options for Simulating Ocean Mesoscale Activity (SOMA) testcases
    [soma]

    # Linear thermal expansion coefficient
    eos_linear_alpha = 0.25
    # Density difference between surface and bottom waters
    density_difference = 5.0
    # Surface temperature value used in initial condition
    surface_temperature = 20.0
    # Surface salinity value used in initial condition
    surface_salinity = 34.0
    # vertical salinity gradient (PSU/m)
    salinity_gradient = 0.0008
    # Depth over which majority of initial stratification is placed (m)
    thermocline_depth = 300
    # Fraction of stratification put into linear profile extending from surface to
    # bottom
    density_difference_linear = 0.05
    # Scale factor controlling width of continental slope. Typically around 0.1
    phi = 0.1
    # Depth of the continental shelf (m)
    shelf_depth = 100
    # Depth of the bottom of the ocean for the SOMA test case (m)
    bottom_depth = 2500

    # the density range and number of surfaces for particles
    min_particle_density = 1028.5
    max_particle_density = 1030.0
    surface_count = 11

The parameters can be altered to change the initial condition or the particle
distribution.

default
-------

``ocean/soma/<resolution>/default`` is the default version of the SOMA
test case from `Wolfram et al. (2015) <https://doi.org/10.1175/JPO-D-14-0260.1>`_.
This test uses 60 unequally spaced layers and does not use surface restoring
or include particle dynamics.

.. image:: images/soma.png
   :width: 500 px
   :align: center

The test case includes a very short (3 time step) test run and validation of
prognostic variables for regression testing.

long
----

Results in Wolfram et al. (2015) are shown from longer simulations, as provided
by the ``ocean/soma/<resolution>/long`` test cases. As in the ``default``
test cases, this test uses 60 unequally spaced layers and does not use surface
restoring or include particle dynamics. Unlike ``default``, this test case is
configured for a longer simulation (3 years) potentially appropriate for
scientific work.

particles
---------

Wolfram et al. (2015) is focused on diffusion as measured by particle
trajectories.  For now, most SOMA test cases do not include particles, but
we include one test case, ``ocean/soma/32km/particles``, with particles from
the Lagrangian, In situ, Global, High-Performance Particle Tracking (LIGHT)
framework for regression testing.  Otherwise, the ``particles`` test case is
identical to ``default``. Particle-relate variables are also included in
the test-case validation.

surface_restoring
-----------------

The ``ocean/soma/<resolution>/surface_restoring`` test cases are identical to
``default`` except that they include restoring of the surface temperature to
a prescribed field that varies linearly with latitude.

three_layer
-----------

The ``ocean/soma/<resolution>/three_layer`` test cases are identical to
``default`` except that they have only 3 vertical layers and do not include the
continental shelf in the simulation domain.