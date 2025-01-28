.. _dev_ocean_dam_break:

dam_break
============

The ``dam_break`` test group
(:py:class:`compass.ocean.tests.dam_break.DamBreak`)
implements variants of the dam break test case.  Here,
we describe the shared framework for this test group and the 1 test case.

.. _dev_ocean_dam_break_framework:

framework
---------

The shared config options for the ``dam_break`` test group are described
in :ref:`ocean_dam_break` in the User's Guide.

Additionally, the test group has shared ``namelist.init`` and 
``namelist.forward`` files with a few common namelist options related to the
dam geometry, run duration, viscosity and wetting-and-drying options, as well
as shared ``streams.init`` and ``streams.forward`` files that defines ``mesh``,
``input``, and ``output`` streams.

initial_state
~~~~~~~~~~~~~

The class
:py:class:`compass.ocean.tests.dam_break.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove periodicity in the x and y directions. MPAS-Ocean is then run
in init mode. During this init mode run, ssh is initialized and ``cullCell`` is
defined to remove cells around the dammed region so that there is one spillway
to the floodplain. The mesh is then culled in python and init mode is run once
more to generate the initial condition on the culled mesh.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.dam_break.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step. The time step is updated depending on the
resolution and MPAS-Ocean is run in ``run()``.

viz
~~~

The class :py:class:`compass.ocean.tests.dam_break.viz.Viz`
defines a visualization step which serves the purpose of validation. This
validation is tailored for the default config options. A multipanel plot
compares the time evolution of ssh at different sites in the domain between
MPAS-Ocean, ROMS, and experimental data. One subplot shows the geometry of the
domain in plan view.


.. _dev_ocean_dam_break_default:

default
-------

The :py:class:`compass.ocean.tests.dam_break.default.Default`
test performs one 10-minute run on 40 cores. It doesn't contain any
:ref:`dev_validation`. This class accepts resolution as an argument. The domain
is configured to have a 1 m by 1 m dammed region matching the experimental
setup and a ~12 m by ~12 m flood plain to minimize reflections off the
boundaries.


.. _dev_ocean_dam_break_ramp:

ramp
----

The :py:class:`compass.ocean.tests.dam_break.ramp.Ramp` is identical to the
default class except it sets ``ramp`` to ``True`` for the forward step to enable
the ramp feature for wetting and drying.
