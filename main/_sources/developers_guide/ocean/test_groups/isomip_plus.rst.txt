.. _dev_ocean_isomip_plus:

isomip_plus
===========

The ``isomip_plus`` test group
(:py:class:`compass.ocean.tests.isomip_plus.IsomipPlus`) implements variants
of the ISOMIP+ experiments (see :ref:`ocean_isomip_plus` in the User's Guide).
Here, we describe the shared framework for this test group and 3 test case
(Ocean0, Ocean1 and Ocean2) that have currently been implemented.

framework
---------

The shared config options for the ``isomip_plus`` test group
are described in :ref:`ocean_isomip_plus` in the User's Guide.

Additionally, the test group has a shared ``namelist.forward`` file with
a few common namelist options related to forcing and analysis members, as well
as a shared ``streams.forward.template`` file that defines ``mesh``, ``input``,
``forcing_data``, ``timeSeriesStatsMonthlyOutput``, ``restart``, ``output``,
``globalStatsOutput``, and ``land_ice_fluxes`` streams.

evap
~~~~

The function :py:func:`compass.ocean.tests.isomip_plus.evap.update_evaporation_flux()`
is used to update the "evaporation" rate and other surface fluxes near the
northern boundary that are used to mimic a spillway, preventing sea level from
rising indefinitely due to the input of freshwater from ice-shelf melting.


geom
~~~~

The function :py:func:`compass.ocean.tests.isomip_plus.geom.interpolate_ocean_mask()`
interpolates the ocean mask from the BISICLES grid of the input geometry to
the MPAS-Ocean mesh.  The mask can later be used to cull land cells from the
MPAS mesh before interpolating other variables to the resulting culled mesh.
Optionally, a thin film under the ice sheet can be used and grounded ice is
not culled.

The function :py:func:`compass.ocean.tests.isomip_plus.geom.interpolate_geom()`
interpolates the remaining geometric variables:

* ``bottomDepthObserved`` -- the bedrock elevation (positive up)

* ``ssh`` -- the sea surface height

* ``oceanFracObserved`` -- the fraction of the mesh cell that is ocean

* ``landIceFraction`` -- the fraction of the mesh cell that is covered
  by an ice shelf

* ``smoothedDraftMask`` -- a smoothed version of the floating mask that
  may be useful for determining where to alter the vertical coordinate
  to accommodate ice-shelf cavities

process_geom
~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.process_geom.ProcessGeom`
defines a step for processing ISOMIP+ geometry before interpolating it to the
MPAS mesh.  This includes applying a simple calving scheme based on a threshold
in ice-shelf thickness, then apply smoothing to the topography data.
Optionally, the ice draft can be scaled by a factor as a simple way to explore
changing ice-shelf topography.  Variables are renamed to those expected by
MPAS-Ocean.

planar_mesh
~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.planar_mesh.PlanarMesh`
defines a step for generating a planar mesh.

cull_mesh
~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.cull_mesh.CullMesh`
defines a step for culling the mesh to include only ocean cells, with or
without a thin film depending on the test.

initial_state
~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.initial_state.InitialState`
defines a step for setting up the initial state for each test case.

First, a mesh appropriate for the resolution is generated using
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()`.  Then, the mesh is
culled to remove land cells.  When a wetting-and-drying algorithm that relies
on a thin film under land ice is used, a distance up to that specified by the
config option ``nx_thin_film`` is retained.  A vertical coordinate is generated,
with 36 layers of 20-m thickness in the open ocean by default.  By default,
the :ref:`dev_ocean_framework_vertical` is ``z-star``, meaning the 1D grid is
"squashed" down so the sea-surface height corresponds to the location of the
ice-ocean interface (ice draft).  The initial temperature and salinity profiles
are computed along with zero initial velocity.  Finally, forcing data fields
are produced for restoring to temperature and salinity profiles at the northern
boundary and for "evaporative" fluxes at the surface that are used to mimic a
spillway, removing water at the northern boundary and preventing runaway
sea-level rise from the the incoming ice-shelf meltwater.

For the time-varying version of a test case, ``initial_state`` also computes
a set of time-varying ``landIcePressureForcing`` and ``landIceDraftForcing``
fields, based on the ``isomip_plus_forcing`` config options (see
:ref:`ocean_isomip_plus_time_varying_ocean0`).  The time evolution of the
``landIcePressure`` and ``landIceDraft`` fields is determined by linear
interpolation in time between consecutive entries in the these forcing
fields, which are stored in a file ``land_ice_forcing.nc``.

Grounding line motion is allowed to occur for a subset of test cases with have
the attribute ``thin_film_present`` set to true. For all other test cases, the
grounding line and calving front are held fixed in time, so the field
``landIceFractionForcing`` is the same as ``landIceFraction`` in the initial
condition for all time.

The ``initial_state`` step also generates horizontal sections through the
domain of layer thicknesses and the mid-layer depth as well as horizontal
sections of initial SSH, land ice presure, and total water column thickness.

ssh_adjustment
~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.ssh_adjustment.SshAdjustment`
performs sea-surface height adjustment described
:ref:`dev_ocean_framework_iceshelf`.  Starting from the initial condition
from ``initial_state``, the test case performs a number of iterations (10 by
default) of forward simulation followed by adjustment of the land-ice pressure
field.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.forward.Forward`
defines ``performance`` and ``simulation`` steps for running MPAS-Ocean from
the initial condition produced in the ``initial_state`` step. A link to the
MPAS-Ocean executable is created when the test case is set up and MPAS-Ocean is
run (including updating PIO namelist options and generating a graph partition)
in ``run()``.

The ``performance`` step is run for only 1 hour (appropriate for regression
testing) except when tidal forcing is applied, in which case the run duration
is 24 hours.  Then, potential temperature and salinity are plotted at the top
and bottom of the ocean and along a cross section of through the middle (y =
40 km) of the domain.

The ``simulation`` step runs for 1 month, then adjusts the "evaporative"
forcing based on the average of the melt fluxes from that month.  Then,
namelist options are modified so the simulation is ready to run for another
month.

See :ref:`ocean_isomip_plus` for a fuller description of how to use the
``performance`` and ``simulation`` steps.

streamfunction
~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.isomip_plus.streamfunction.Streamfunction`
defines a step for computing the barotropic (vertically integrated) and
overturning streamfunctions from the latest simulation results from the
``simulation`` step.  This step is intended to be run repeatedly each time new
simulation results come in, but can also be run once at the end of a longer
simulation.

viz
~~~

The :py:class:`compass.ocean.tests.isomip_plus.viz.Viz` class defines a step
for performing visualization of ISOMIP+ results.  This step should be run
after running ``simulation`` any number of times and then ``streamfunction``
(unless you set ``plot_streamfunctions = False`` in the ``[isomip_plus_viz]``
section of the config file).  Movie frames an time series plots will appear
in the ``plots`` directory; The movies themselves in ``movies``, and some
time series averaged only over the deepest parts of the ice draft in
``timeSeriesBelow300m``.

misomip
~~~~~~~

The :py:class:`compass.ocean.tests.isomip_plus.misomip.Misomip` class defines
a step for interpolating the results to the standard MISOMIP grid and writing
out the results in the format expected by MISOMIP.

.. note::

    There is currently an issue with fill values not being handled correctly
    that needs to be resolved before this step is fully useful.

.. _dev_ocean_isomip_plus_test:

isomip_plus_test
----------------

The same class,
:py:class:`compass.ocean.tests.isomip_plus.isomip_plus_test.IsomipPlusTest`,
defines the Ocean0, Ocean1 and Ocean2 test cases at various resolutions and with
various vertical coordinates.  By default, these test cases only run 3 of the
7 available steps: ``initial_state`` to create and mesh and initial condition,
``ssh_adjustment`` to perform 10 1-hour simulations used to balance the
land-ice pressure with the sea surface height, and ``performance`` to run a
final 1-hour (15-time-step) forward simulation. If a baseline is provided when
calling :ref:`dev_compass_setup`, a large number of variables (both prognostic
and related to land-ice fluxes) are checked to make sure they match the
baseline.

The optional ``simulation``, ``streamfunction``, ``viz`` and ``misomip`` steps,
described above, are used to perform longer simulations and perform analysis
and visualization.
