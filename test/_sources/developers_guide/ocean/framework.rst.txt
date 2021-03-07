.. _dev_ocean_framework:

Ocean framework
===============

The ``ocean`` core contains a small amount of shared framework code that we
expect to expand in the future as new test cases are added.

Namelist replacements and streams files
---------------------------------------

The modules ``compass.ocean.namelists`` and ``compass.ocean.streams`` contain
namelist replacements and streams files that are similar to core-level
templates in :ref:`legacy_compass`.  Current templates are for adjusting sea
surface height in ice-shelf cavities, and outputting variables related to
frazil ice and land-ice fluxes.

.. _dev_ocean_framework_vertical:

Vertical coordinate
-------------------

The ``compass.ocean.vertical`` module provides support for 1D vertical
coordinates and the 3D `z* coordinate <https://doi.org/10.1016/j.ocemod.2003.09.003>`_.

To create 1D vertical grids, test cases should call
:py:func:`compass.ocean.vertical.generate_grid()` with the desired config
options set in the ``vertical_grid`` section (as described in :ref:`ocean_vertical`).

The function :py:func:`compass.ocean.vertical.zstar.compute_layer_thickness_and_zmid()`
can be used to compute ``layerThickness``, ``zMid`` and ``restingThickness``
variables for a z* vertical coordinate based on a 1D vertical grid and
bathymetry.

.. _dev_ocean_framework_iceshelf:

Ice-shelf cavities
------------------

The module ``compass.ocean.iceshelf`` defines two functions that are used to
set up domains with ice-shelf cavities.

:py:func:`compass.ocean.iceshelf.compute_land_ice_pressure_and_draft()`
computes the ``landIcePressure`` and ``landIceDraft`` fields based on the
sea-surface height (SSH) and a reference density (typically the the Boussinesq
reference density).

:py:func:`compass.ocean.iceshelf.adjust_ssh()` performs a series of forward
runs with MPAS-Ocean to detect and correct imbalances between the SSH and the
land-ice pressure.  In each forward run, the SSH is allowed to evolve forward
in time for a short period (typically 1 hour), then the resulting change in
SSH is translated into a compensating change in land-ice pressure that is
expected to reduce the change in SSH.  The initial land-ice pressure is updated
accordingly and the process is repeated for a fixed number of iterations,
typically leading to smaller and smaller changes in the land-ice pressure.
This process does not completely eliminate the dynamical adjustment of the
ocean to the overlying weight of the ice shelf but it tends to reduce it
substantially and to prevent it from causing numerical instabilities.  This
procedure is also largely agnostic to the equation of state being used or the
method for implementing the horizontal pressure-gradient force.

.. _dev_ocean_framework_particles:

Particles
---------

The ``compass.ocean.particles`` module contains functionality for initializing
particles for the `LIGHT framework <https://doi.org/10.1175/JPO-D-14-0260.1>`_.

:py:func:`compass.ocean.particles.write()` creates an initial condition for
particles partitioned across cores.  There are 3 possible particle types (or
``all`` to indicate that all 3 types will be generated):

``buoyancy``
  Particles are constrained to buoyancy (isopycnal) surfaces

``passive``
  Particles move both horizontally and vertically as passive tracers

``surface``
  Particles are constrained to the top ocean level

:py:func:`compass.ocean.particles.remap_particles()` is used to remap particles
onto a new grid decomposition.  This might be useful, for example, if you wish
to change the number of cores that a particle initial condition should run on.

.. _dev_ocean_framework_plot:

Plotting
--------

The ``compass.ocean.plot`` contains functionality for plotting the initial
state and 1D vertical grid.

:py:func:`compass.ocean.plot.plot_initial_state()` creates histogram plots of
salinity, temperature, bottom depth, ``maxLevelCell``, layer thickness and the
Haney number from global initial condition.  This is useful for providing a
quick sanity check that these values have the expected range and distribution,
based on previous meshes.

:py:func:`compass.ocean.plot.plot_vertical_grid()` plot the vertical grid in
3 ways: layer mid-depth vs. vertical index; layer mid-depth vs. layer thickness;
and layer thickness vs. vertical index.  Again, this provides a quick sanity
check that the grid has the expected bounds (both in thickness and in depth)
and number of layers.
