.. _ocean_isomip_plus:

isomip_plus
===========

The ``ocean/isomip_plus`` test group includes variants of the Ice Sheet-Ocean
Model Intercomparison Project, second phase (ISOMIP+) experiments from
`Asay-Davis et al. (2016) <https://doi.org/10.5194/gmd-9-2471-2016>`_.  These
experiments use `idealized ice-shelf geometry <https://doi.org/10.5880/PIK.2016.002>`_
from the Marine Ice SheetModel Intercomparison Project, third phase (MISMIP+;
see `Cornford et al. 2020 <https://doi.org/10.5194/tc-14-2283-2020>`_)
performed with the BISICLES ice-sheet model.

Currently, only the Ocean0 experiment is supported but the plan is to add the
Ocean1 and Ocean2 experiments in the next few months, and the Ocean3 and Ocean4
experiments at a later date, once MPAS-Ocean supports moving grounding lines.

By default, the test case is available at 2 km and 5 km horizontal resolution
with a z-star :ref:`dev_ocean_framework_vertical`.  The test case has 36
vertical layers, each of 20-m thickness outside of the ice-shelf cavity.

The initial temperature for the whole domain is constant (1 degree Celsius),
while salinity varies linearly with depth from 34.5 PSU at the sea surface
to 34.7 PSU at the sea floor, which is at a constant at 2000 m depth.  The
conceptual overlying ice shelf depresses the sea surface height buy as much as
1990 m (leaving a 10-m water column) for the first 30 km in y.  Over the next
30 km, it rises to 1490 m, then fairly abruptly to zero over the next 15 km,
where it remains for the second half of the domain.  The ice shelf occupies
these first 75 km of the domain: fluxes from ice-shelf melting are only applied
in this region.

.. figure:: images/isomip_plus.png
   :width: 500 px
   :align: center

   A cross section through the center (y = 40 km) of the ISOMIP+ Ocean0 test
   case at 5 km resolution, showing potential temperature averaged over month
   9 of the simulation.

The ``isomip_plus`` test cases are composed of 5 steps that run by default:
``process_geom``, which loads the data on the ice sheet mesh; ``planar_mesh``,
which defines the planar mesh; ``cull_mesh``,  which culls the mesh;
``initial_state``, which interpolates the ice geometry and computes the
initial conditions for the model; ``ssh_adjustment``, which modifies the
``landIcePressure`` field to balance the ``ssh`` field, see
:ref:`ocean_ssh_adjustment`; and ``performance``, which performs a 1-hour time
integration of the model and compares the results with a baseline if one is
provided.

Four additional steps can optionally be run: ``simulation``, which performs
one month of simulation, then updates the "evaporative" fluxes used in the test
case to prevent sea level from rising significantly due to meltwater inflow at
the ice-shelf base; ``streamfunction``, which computes the barotropic
(vertically integrated) and overturning streamfunctions; ``viz``, which plots
time series and movies of various variables of interest; and ``misomip``, which
interpolates the results to the MISOMIP comparison grid.

shared config options
---------------------

The ``isomip_plus`` test cases share the following config options:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = uniform

    # Number of vertical levels
    vert_levels = 36

    # Depth of the bottom of the ocean
    bottom_depth = 720.0

    # The type of vertical coordinate (e.g. z-level, z-star)
    coord_type = z-star

    # Whether to use "partial" or "full", or "None" to not alter the topography
    partial_cell_type = None

    # The minimum fraction of a layer for partial cells
    min_pc_fraction = 0.1

    # Options relate to adjusting the sea-surface height or land-ice pressure
    # below ice shelves to they are dynamically consistent with one another
    [ssh_adjustment]

    # the number of iterations of ssh adjustment to perform
    iterations = 10


    # config options for ISOMIP+ test cases
    [isomip_plus]

    # number of cells over which to smooth topography
    topo_smoothing = 1.0

    # minimum thickness of the ice shelf, below which it is removed ("calved")
    min_ice_thickness = 100.0

    # a scalar by which the ice draft will be scaled (squashed).  This is
    # convenient for testing vertical coordinates
    draft_scaling = 1.0

    # Minimum number of vertical levels in a column
    minimum_levels = 3

    # The minimum allowable layer thickness
    min_layer_thickness = 0.0

    # Minimum thickness of the initial ocean column (to prevent 'drying')
    min_column_thickness = 10.0

    # Minimum fraction of a cell that contains ocean (as opposed to land or
    # grounded land ice) in order for it to be an active ocean cell.
    min_ocean_fraction = 0.5

    # Threshold used to determine how far from the ice-shelf the sea-surface height
    # can be adjusted to keep the Haney number under control
    min_smoothed_draft_mask = 0.01

    # Minimum fraction of a cell that contains land ice in order for it to be
    # considered a land-ice cell by MPAS-Ocean (landIceMask == 1).
    min_land_ice_fraction = 0.5

    # the initial temperature at the sea surface
    init_top_temp = -1.9
    # the initial temperature at the sea floor
    init_bot_temp = 1.0
    # the initial salinity at the sea surface
    init_top_sal = 33.8
    # the initial salinity at the sea floor
    init_bot_sal = 34.7

    # the restoring temperature at the sea surface
    restore_top_temp = -1.9
    # the restoring temperature at the sea floor
    restore_bot_temp = 1.0
    # the restoring salinity at the sea surface
    restore_top_sal = 33.8
    # the restoring salinity at the sea floor
    restore_bot_sal = 34.7

    # restoring rate (1/days) at the open-ocean boundary
    restore_rate = 10.0

    # the "evaporation" rate  (m/yr) near the open-ocean boundary used to keep sea
    # level from rising
    restore_evap_rate = 200.0

    # southern boundary of restoring region (m)
    restore_xmin = 790e3
    # northern boundary of restoring region (m)
    restore_xmax = 800e3

    # Coriolis parameter (1/s) for entire domain
    coriolis_parameter = -1.409e-4

    # initial value for the effective density (kg/m^3) of seawater for entire
    # domain
    effective_density = 1026.

    # config options for ISOMIP+ time-varying land-ice forcing
    [isomip_plus_forcing]

    # the forcing dates
    dates = 0001-01-01_00:00:00, 0002-01-01_00:00:00, 0003-01-01_00:00:00

    # the amount by which the initial landIcePressure and landIceDraft are scaled
    # at each date
    scales = 0.1, 1.0, 1.0

    # config options for computing ISOMIP+ streamfunctions
    [isomip_plus_streamfunction]

    # the resolution of the overturning streamfunction in m
    osf_dx = 2e3
    osf_dz = 5.

    # config options for visualizing ISOMIP+ ouptut
    [isomip_plus_viz]

    # whether to plot the Haney number
    plot_haney = True

    # whether to plot the barotropic and overturning streamfunctions
    plot_streamfunctions = True

    # frames per second for movies
    frames_per_second = 30

    # movie format
    movie_format = mp4

    # the y value at which a cross-section is plotted (in m)
    section_y = 40e3

You can modify the horizontal mesh, vertical grid, geometry, and initial
temperature and salinity of the test case by altering these options.

.. _ocean_isomip_plus_ocean0:

Ocean0
------

``ocean/isomip_plus/2km/z-star/Ocean0`` and
``ocean/isomip_plus/5km/z-star/Ocean0``

This test case is initialized with "warm" ocean conditions: 1 degree C at the
sea floor, decreasing to -1.9 degrees C at the ocean surface.  These conditions
are approximately similar to those in the warmest waters on the Antarctic
continental shelf in the Amundsen and Bellingshausen Seas.  At the northern
boundary, the temperature is restored to the same warm profile, leading to a
vigorous circulation under the ice shelf that continually supplies heat and
produces relatively high melt rates.  Because of the rigorous flow, the
simulation reaches a quasi-steady state in 2-3 years.

.. _ocean_isomip_plus_ocean1:

Ocean1
------

``ocean/isomip_plus/2km/z-star/Ocean1`` and
``ocean/isomip_plus/5km/z-star/Ocean1``

This test case is initialized with "cold" ocean conditions: -1.9 degree C
throughout the water column.  These conditions are similar to cold-shelf
regions such as the Antarctic continental shelf in the Weddell and Ross Seas.
At the northern boundary, the temperature is restored to the same warm profile
as in Ocean0.  The initially cold cavity has low melt rates and a weak flow, so
that warm water from the northern boundary may take about a decade to reach the
ice-shelf base.  At this point, the melting and flow rapidly increase,
eventually (in the coarse of ~20 years) leading to the same quasi-steady-state
as in Ocean0.  The ISOMIP+ protocol suggests running this simulation for 20
years.

.. _ocean_isomip_plus_ocean2:

Ocean2
------

``ocean/isomip_plus/2km/z-star/Ocean2`` and
``ocean/isomip_plus/5km/z-star/Ocean2``

This test case is initialized with "warm" ocean conditions as in Ocean0.  At
the northern boundary, the temperature is restored to the cold profile used for
the initial condition in Ocean1: -1.9 degree C throughout the water column.
Thus, where Ocean1 transitions from cold to warm cavity conditions, Ocean2
makes the opposite transition from warm to cold.  The geometry is also taken
from a different stage of the BISICLES MISIMP+ run than Ocean0 and Ocean1 in
which the ice shelf has undergone significant thinning and retreat.  The
initially warm cavity has high melt rates and a strong flow, so that cold water
water from the northern boundary will reach the ice-shelf base within a few
years.  At this point, the melting and flow exponentially decrease, approaching
a new quasi-steady state.  The ISOMIP+ protocol suggests running this
simulation for 20 years, which is not long enough to reach quasi-steady state.

.. _ocean_isomip_plus_time_varying_ocean0:

time_varying_Ocean0
-------------------

``ocean/isomip_plus/2km/z-star/time_varying_Ocean0`` and
``ocean/isomip_plus/5km/z-star/time_varying_Ocean0``

This test case is identical to ``Ocean0`` except that the land-ice pressure
and land-ice draft are prescribed to evolve in a very simple way in time.
By default, the these 2 fields start out at year 0001 with 10% of their normal
value (so the ice shelf is 10% of its thickness in a normal ``Ocean0`` run).
Then, over the course of a year, both fields increase to 100% of their normal
value and stay there for another year.  This test case is a simple way of
exploring changing ice thickness without the need to support a changing
grounding line (which remains fixed in time).

Users can modify the test case by adding or modifying entries in these config
options before running the test case:

.. code-block:: cfg

    # config options for ISOMIP+ time-varying land-ice forcing
    [isomip_plus_forcing]

    # the forcing dates
    dates = 0001-01-01_00:00:00, 0002-01-01_00:00:00, 0003-01-01_00:00:00

    # the amount by which the initial landIcePressure and landIceDraft are scaled
    # at each date
    scales = 0.1, 1.0, 1.0

Dates do not have to be the beginnings of years, they could be any list that is
monotonic in time. Scales can be any fraction between 0.0 and 1.0.

thin_film_Ocean0
----------------

``ocean/isomip_plus/2km/z-star/thin_film_Ocean0`` and
``ocean/isomip_plus/5km/z-star/thin_film_Ocean0``

The thin-film version of Ocean0 turns the wetting-and-drying scheme on in
MPAS-Ocean and features a thin ocean layer below the grounded ice of thickness
``min_column_thickness`` specified in the config file. In the non-time-varying
version of this test case, the behavior should be the same as the version
without a thin film (``Ocean0``).

There are also several time-varying versions of this test case:
``ocean/isomip_plus/${RES}/${COORD}/thin_film_time_varying_Ocean0``,
``ocean/isomip_plus/${RES}/${COORD}/thin_film_wetting_Ocean0``, and
``ocean/isomip_plus/${RES}/${COORD}/thin_film_drying_Ocean0``. The
latter two prescribe decreasing or increasing land ice pressure, respectively,
to simulate grounding line motion in the landward or seaward directions. The
resolutions supported (``RES``) are ``2km`` and ``5km`` and the coordinate
types (``COORD``) are ``sigma`` and ``single_layer``.

thin_film_tidal_forcing_Ocean0
------------------------------

``ocean/isomip_plus/2km/single_layer/thin_film_tidal_forcing_Ocean0`` and
``ocean/isomip_plus/5km/single_layer/thin_film_tidal_forcing_Ocean0``

The tidal forcing test case uses the existing tidal boundary forcing in the
forward mode of MPAS-Ocean to drive SSH variations in the far-field that
propagate into the ice shelf cavity. Given the geometry of the Ocean0 test
case, these tidal SSH variations should not produce any grounding line motion.
Thus, this is a test of the robustness of the wetting-and-drying algorithm to
small pressure perturbations.

Performance run
---------------

By default, ``isomip_plus`` test cases are configured for "performance" runs.
The initial condition is created, the the sea surface height and ice-shelf
pressure are adjusted to be in balance.  Then, a simulation is performed for
only 1 simulated hour (appropriate for regression testing).  For the tidally-
varying case, the simulation is extended to 24 hours but is still
computationally inexpensive due to the single-layer configuration. Finally,
potential temperature and salinity are plotted at the top and bottom of the
ocean and along a cross section of through the middle (y = 40 km) of the
domain.

Simulation run
--------------

``isomip_plus`` test cases can be manually configured for longer simulation
runs.  First, do a performance run as described above (the default when you
just do ``compass run`` in the test case work directory).

Then, edit the config file in the work directory (e.g. ``Ocean0.cfg``) to set
``setup_to_run = simulation streamfunction viz`` in the ``[test_case]`` section
at the very top.  With this setting, one month of simulation will be performed,
then the streamfunctions will be computed based on the latest results in the
``streamfunction`` step and time series plots and movies will be updated in
the ``viz`` step.  You can manually keep running ``compass run`` in the test
case directory to run a month at a time, or you can create a job script to
run ``compass run`` repeatedly (say 240 times for a 20-year simulation) inside
a for-loop.
