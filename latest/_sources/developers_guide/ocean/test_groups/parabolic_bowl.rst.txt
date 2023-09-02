.. _dev_ocean_parabolic_bowl:

parabolic_bowl
==================

The ``parabolic_bowl`` test group
(:py:class:`compass.ocean.tests.parabolic_bowl.ParabolicBowl`)
implements convergence study for wetting and drying in a parabolic bowl.

.. _dev_ocean_parabolic_bowl_default:

default
-------

The :py:class:`compass.ocean.tests.parabolic_bowl.default.Default`
test performs a series of 3 day-long runs where an initial mound of water
oscillates in a parabolic bowl.  The resolution of the problem (by default, between
5 and 20 km).  Modeled results are compared with a known exact solution to
demonstrate convergence.  See :ref:`ocean_parabolic_bowl_default`.
for config options and more details on the test case.

init
~~~~

The class :py:class:`compass.ocean.tests.parabolic_bowl.init.Init`
defines a step for setting up for generating planar hex meshes and creating
the initial state for each test case.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.parabolic_bowl.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step is determined from the resolution
based on the ``dt_per_km`` config option.  Other namelist options are taken
from the test case's ``namelist.forward``.

viz
~~~

The class :py:class:`compass.ocean.tests.parabolic_bowl.viz.Viz`
defines a step for creating contour plots of the solution at different 
times (``solution_***.png``), time series plots at different points
(``points.png``). It also computes the root mean squared error for 
the results at each resolution and plots them in ``error.png``.
