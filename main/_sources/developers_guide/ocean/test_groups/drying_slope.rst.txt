.. _dev_ocean_drying_slope:

drying_slope
============

The ``drying_slope`` test group
(:py:class:`compass.ocean.tests.drying_slope.DryingSlope`)
implements variants of the drying slope test case.  Here,
we describe the shared framework for this test group and the 1 test case.

.. _dev_ocean_drying_slope_framework:

framework
---------

The shared config options for the ``drying_slope`` test group are described
in :ref:`ocean_drying_slope` in the User's Guide.

Additionally, the test group has shared ``namelist.init`` and
``namelist.forward`` files with a few common namelist options related to run
duration, bottom drag, and tidal forcing options, as well as shared
``streams.init`` and ``streams.forward`` files that defines ``mesh``, ``input``,
``restart``, ``forcing`` and ``output`` streams.

Namelist options specific vertical coordinates are given in
``namelist.${COORD}*`` files.

initial_state
~~~~~~~~~~~~~

The class
:py:class:`compass.ocean.tests.drying_slope.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction. The vertical grid is
configured according to the config options in the ``vertial_grid`` section.
The bottom depth is then set according to config options
``right_bottom_depth`` and ``left_bottom_depth``. The initial layer
thicknesses are set to the minimum thickness and the initial state is set to
a constant value by default according to the config options
``background_temperature`` and ``background_salinity``. Optionally, a mass of
warmer water can be initialized over part of the domain, a "plug," using
config options ``plug_width_frac`` and ``plug_width_temperature``. For current
configurations, tracer tendencies are off and these tracer options do not
affect the flow.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.drying_slope.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. If ``damping_coeff`` is provided as an argument to
the constructor, the associate namelist option
(``config_Rayleigh_damping_coeff``) will be given this value. MPAS-Ocean is run
in ``run()``. The time step is determined as a function of resolution by the
config option ``dt_per_km``. The number of tasks is determined as a function
of resolution by ``ntasks_baseline`` and ``min_tasks``.

viz
~~~

The class :py:class:`compass.ocean.tests.drying_slope.viz.Viz`
defines a visualization step which serves the purpose of validation. This
validation is tailored for the default config options and the two Rayleigh
damping coefficients set by the default sigma-coordinate test case, 0.0025 and
0.01. One plot verifies that the time evolution of the ssh forcing at the
boundary matches the analytical solution intended to drive the test case.
Another plot compares the time evolution of the ssh profile across the channel
between the analytical solution, MPAS-Ocean and ROMS. Similar plots are used
to create a movie showing the solution from MPAS-Ocean at more fine-grained
time intervals.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.drying_slope.analysis.Analysis`
produces a convergence plot for a series of forward steps at different
resolutions. It uses the analytical solution available at 5 discrete times t
compute the RMSE.

.. _dev_ocean_drying_slope_default:

default
-------

The :py:class:`compass.ocean.tests.drying_slope.default.Default`
test performs two 12-hour runs on 4 cores. It doesn't contain any
:ref:`dev_validation`. This class accepts resolution and coordinate type
``coord_type`` as arguments. Both ``sigma`` and ``single_layer`` coordinate
types are supported. For ``sigma`` coordinates, this case is hard-coded to run
two cases at different values of ``config_Rayleigh_damping_coeff``, 0.0025 and
0.01, for which there is comparison data. The ``single_layer`` case runs at one
value of the implicit bottom drag coefficient.

.. _dev_ocean_drying_slope_convergence:

convergence
-----------

The :py:class:`compass.ocean.tests.drying_slope.convergence.Convergence` expands
on the default class to include initial and forward steps for multiple
resolutions and an analysis step to generate a convergence plot.

.. _dev_ocean_drying_slope_decomp:

decomp
------

The :py:class:`compass.ocean.tests.drying_slope.decomp.Decomp`
test performs two 12-hour runs on 1 and 12 cores, respectively.
:ref:`dev_validation` is performed by comparing the output of the two runs.
This class accepts resolution and coordinate type ``coord_type`` as arguments.
Both ``sigma`` and ``single_layer`` coordinate types are supported. For
``sigma`` coordinates, this case is hard-coded to run with
``config_Rayleigh_damping_coeff`` equal to 0.01. The ``single_layer`` case
runs at one value of the implicit bottom drag coefficient.


loglaw
------

The :py:class:`compass.ocean.tests.drying_slope.loglaw.LogLaw` is identical to the
default class except it uses the log-law implicit drag option.
