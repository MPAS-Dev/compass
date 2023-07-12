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

Namelist options specific to resolutions and vertical coordinates are given in
``namelist.${RES}.*`` and ``namelist.${COORD}*`` files.

initial_state
~~~~~~~~~~~~~

The class
:py:class:`compass.ocean.tests.drying_slope.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the y direction. MPAS-Ocean is then run in init
mode. If the vertical coordinate is ``sigma``, the vertical grid configured
accordingly during init mode. During this init mode run, ssh is initialized
using the tidal forcing config options, and temperature and salinity are set
to constant values by default. (Namelist options may be modified to produce a
plug of different temperature values from the background, but this is not
employed in this test case.)

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.drying_slope.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. If ``damping_coeff`` is provided as an argument to 
the constructor, the associate namelist option
(``config_Rayleigh_damping_coeff``) will be given this value. MPAS-Ocean is run
in ``run()``.

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


.. _dev_ocean_drying_slope_ramp:

ramp
----

The :py:class:`compass.ocean.tests.drying_slope.ramp.Ramp` is identical to the
default class except it sets ``ramp`` to ``True`` for the forward step to enable
the ramp feature for wetting and drying.


.. _dev_ocean_drying_slope_log_law:

loglaw
------

The :py:class:`compass.ocean.tests.drying_slope.loglaw.LogLaw` is identical to the
default class except it uses the log-law implicit drag option.
