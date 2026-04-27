.. _landice_framework:

Land-ice Framework
==================


extrapolate
~~~~~~~~~~~

The landice framework module ``compass/landice/extrapolate.py`` provides a
function for extrapolating variables into undefined regions.  It is copied
from a similar script in MPAS-Tools.

mesh
~~~~

The config options below should be defined for all ``mesh_gen`` test cases
(although not necessarily with the same values shown here, which are the
defaults for the 1–10km Humboldt mesh).

A core set of options is **always required**: ``levels``,
``x_min``/``x_max``/``y_min``/``y_max``, ``cull_distance``, ``min_spac``,
``max_spac``, and the five density-function toggles ``use_speed``,
``use_dist_to_edge``, ``use_dist_to_grounding_line``, ``use_dist_to_coast``,
and ``use_bed``.

The remaining mesh-density parameters are **only required when their
corresponding toggle is enabled**, so configs that disable a density function
may omit the associated options entirely:

* ``use_speed = True`` requires ``high_log_speed`` and ``low_log_speed``.
* ``use_dist_to_edge = True`` or ``use_dist_to_grounding_line = True``
  requires ``high_dist`` and ``low_dist``.
* ``use_dist_to_coast = True`` requires ``high_dist_coast`` and
  ``low_dist_coast``.
* ``use_bed = True`` requires ``high_dist_bed``, ``low_dist_bed``,
  ``high_bed``, and ``low_bed``.

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

    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = False
    use_dist_to_edge = True
    # When use_dist_to_coast = True, also set high_dist_coast and low_dist_coast
    use_dist_to_coast = False
    use_bed = True

    # Required when use_speed = True
    # log10 of max speed (m/yr) for cell spacing
    high_log_speed = 2.5
    # log10 of min speed (m/yr) for cell spacing
    low_log_speed = 0.75

    # Required when use_dist_to_edge = True or
    # use_dist_to_grounding_line = True
    # distance at which cell spacing = max_spac (meters)
    high_dist = 1.e5
    # distance within which cell spacing = min_spac (meters)
    low_dist = 1.e4

    # Required when use_dist_to_coast = True
    # distance at which cell spacing in ocean = max_spac (meters)
    high_dist_coast = 16.e3
    # distance within which cell spacing in ocean = min_spac (meters)
    low_dist_coast = 8.e3

    # Required when use_bed = True
    # distance at which bed topography has no effect
    high_dist_bed = 1.e5
    # distance within which bed topography has maximum effect
    low_dist_bed = 5.e4
    # Bed elev beneath which cell spacing is minimized
    low_bed = 50.0
    # Bed elev above which cell spacing is maximized
    high_bed = 100.0
