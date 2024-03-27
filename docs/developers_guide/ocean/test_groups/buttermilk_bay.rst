.. _dev_ocean_buttermilk_bay:

buttermilk_bay
==================

The ``buttermilk_bay`` test group
(:py:class:`compass.ocean.tests.buttermilk_bay.ButtermilkBay`)
implements a realistic regional test for wetting and drying in
a multi-bay system.

.. _dev_ocean_buttermilk_bay_default:

default
-------

The :py:class:`compass.ocean.tests.buttermilk_bay.default.Default`
test performs a series of 2 day-long runs where an initially still bay
is forced at the boundary with a diurnal tidal signal.
The resolution of the problem (by default, between 64 and 256 m) is 
varied to demonstrate the accuracy improvement of a given wetting 
and drying approach. See :ref:`ocean_buttermilk_bay_default`.
for config options and more details on the test case.

init
~~~~

The class :py:class:`compass.ocean.tests.buttermilk_bay.init.Init`
defines a step for setting up for generating planar hex meshes and creating
the initial state for each test case.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.buttermilk_bay.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step is determined from the resolution
based on the ``dt_per_m`` config option.  Other namelist options are taken
from the test case's ``namelist.forward``.

viz
~~~

The class :py:class:`compass.ocean.tests.buttermilk_bay.viz.Viz`
defines a step for creating contour plots of the solution at different 
times (``solution_***.png``), and time series plots at different points
(``points.png``)
