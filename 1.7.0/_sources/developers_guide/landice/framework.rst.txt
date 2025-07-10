.. _dev_landice_framework:

Land-ice Framework
==================

ais_observations
~~~~~~~~~~~~~~~~

The landice framework module ``compass/landice/ais_observations.py`` contains
observational data for various Antarctic basins.  These are based on the
ISMIP6 basin definitions, but it need not be limited to those.  The basins do
not need to be mutually exclusive, so more can be added as needed.

extrapolate
~~~~~~~~~~~

The landice framework module ``compass/landice/extrapolate.py`` provides a
function for extrapolating variables into undefined regions.  It is copied
from a similar script in MPAS-Tools.

iceshelf_melt
~~~~~~~~~~~~~
The landice framework module ``compass/landice/iceshelf_melt.py`` provides
functionality related to ice-shelf basal melting.  Currently, there is a
single function ``calc_mean_TF`` that calculated the mean thermal forcing
along the ice-shelf draft in a domain for a given geometry file and thermal
forcing file.

mesh
~~~~

The landice framework module :py:mod:`compass.landice.mesh` provides functions
used by all ``mesh_gen`` tests cases, which currently exist within the
``antarctica``, ``crane``, ``greenland``, ``humboldt``, ``kangerlussuaq``,
``koge_bugt_s``, and ``thwaites`` test groups. These functions include:

:py:func:`compass.landice.mesh.add_bedmachine_thk_to_ais_gridded_data()`
copies BedMachine thickness to the AIS reference gridded dataset.
It replaces thickness field in the compilation dataset with the one we
will be using from BedMachine for actual thickness interpolation
There are significant inconsistencies between the masking of the two,
particularly along the Antarctic Peninsula, that lead to funky
mesh extent and culling if we use the thickness from 8km composite
dataset to define the cullMask but then actually interpolate thickness
from BedMachine.
This function uses bilinear interpolation to interpolate from the 500 m
resolution of BedMachine to the 8 km resolution of the reference dataset.
It is not particularly accurate, but is fast and adequate for generating
the flood filled mask for culling the mesh.  Highly accurate conservative
remapping is performed later for actually interpolating BedMachine
thickness to the final MALI mesh.

:py:func:`compass.landice.mesh.clean_up_after_interp()` performs some final
clean up steps after interpolation for the AIS mesh case.

:py:func:`compass.landice.mesh.gridded_flood_fill()` applies a flood-fill algorithm
to the gridded dataset in order to separate the ice sheet from peripheral ice.

:py:func:`compass.landice.mesh.interp_gridded2mali()` interpolates gridded data
(e.g. BedMachine thickness or MEaSUREs ice velocity) to a MALI mesh, accounting
for masking of the ice extent to avoid interpolation ramps. This functions works
for both Antarctica and Greenland. 

:py:func:`compass.landice.mesh.preprocess_ais_data()` performs adjustments to
gridded AIS datasets needed for rest of compass workflow to utilize them.

:py:func:`compass.landice.mesh.set_rectangular_geom_points_and_edges()` sets node
and edge coordinates to pass to py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.

:py:func:`compass.landice.mesh.set_cell_width()` sets cell widths based on settings
in config file to pass to :py:func:`mpas_tools.mesh.creation.build_mesh.build_planar_mesh()`.
Requires the following options to be set in the given config section: ``min_spac``,
``max_spac``, ``high_log_speed``, ``low_log_speed``, ``high_dist``, ``low_dist``,
``high_dist_bed``, ``low_dist_bed``, ``high_bed``, ``low_bed``, ``cull_distance``,
``use_speed``, ``use_dist_to_edge``, ``use_dist_to_grounding_line``, and ``use_bed``.

:py:func:`compass.landice.mesh.get_dist_to_edge_and_gl()` calculates distance from
each point to ice edge and grounding line, to be used in mesh density functions in
:py:func:`compass.landice.mesh.set_cell_width()`. In future development,
this should be updated to use a faster package such as `scikit-fmm`.

:py:func:`compass.landice.mesh.build_cell_width()` determine final MPAS mesh cell sizes
using desired cell widths calculated by py:func:`compass.landice.mesh.set_cell_width()`,
based on user-defined density functions and config options.

:py:func:`compass.landice.mesh.build_mali_mesh()` creates the MALI mesh based on final
cell widths determined by py:func:`compass.landice.mesh.build_cell_width()`, using Jigsaw
and MPAS-Tools functions. Culls the mesh based on config options, interpolates
all available fields from the gridded dataset to the MALI mesh using the bilinear
method, and marks domain boundaries as Dirichlet cells.

:py:func:`compass.landice.mesh.make_region_masks()` creates region masks using regions
defined in Geometric Features repository. It is only used by the ``antarctica``
and ``greenland`` test cases.

The following config options should be defined for all ``mesh_gen`` test cases (although
not necessarily with the same values shown here, which are the defaults for the 1–10km
Humboldt mesh):

.. code-block:: cfg

    # config options for humboldt test cases
    [mesh]

    # number of levels in the mesh
    levels = 10

    # Bounds of Humboldt domain. If you want the extent
    # of the gridded dataset to determine the extent of
    # the MALI domain, set these to None.
    x_min = -630000.
    x_max = 84000.
    y_min = -1560000.
    y_max = -860000.

    # distance from ice margin to cull (km).
    # Set to a value <= 0 if you do not want
    # to cull based on distance from margin.
    cull_distance = 5.0

    # mesh density parameters
    # minimum cell spacing (meters)
    min_spac = 1.e3
    # maximum cell spacing (meters)
    max_spac = 1.e4
    # log10 of max speed (m/yr) for cell spacing
    high_log_speed = 2.5
    # log10 of min speed (m/yr) for cell spacing
    low_log_speed = 0.75
    # distance at which cell spacing = max_spac (meters)
    high_dist = 1.e5
    # distance within which cell spacing = min_spac (meters)
    low_dist = 1.e4

    # These *_bed settings are only applied when use_bed = True.
    # distance at which bed topography has no effect
    high_dist_bed = 1.e5
    # distance within which bed topography has maximum effect
    low_dist_bed = 5.e4
    # Bed elev beneath which cell spacing is minimized
    low_bed = 50.0
    # Bed elev above which cell spacing is maximized
    high_bed = 100.0

    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = False
    use_dist_to_edge = True
    use_bed = True
