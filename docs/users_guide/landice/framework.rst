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
