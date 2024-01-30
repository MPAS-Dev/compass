.. _landice_isunnguata_sermia:

isunnguata_sermia
=================

The ``landice/isunnguata_sermia`` test group includes a test case for creating a
mesh for Isunnguata Sermia, Greenland. The optimization for basal friction
happens outside of COMPASS because it requires expert usage and takes a
larger amount of computing resources than COMPASS is typically run with.

.. figure:: images/isunnguata_sermia_1to10km.png
   :width: 777 px
   :align: center

   Ice thickness in meters on Isunnguata Sermia 1-10km variable resolution mesh.

The test group includes a single test case that creates the variable resolution mesh.

config options
--------------

The test group uses the following default config options.  At this point only
the mesh generation options are adjusted through the config file.

.. code-block:: cfg

    # config options for isunnguata sermia test cases
    [mesh]
    
    # number of levels in the mesh
    levels = 10
    
    # Bounds of isunnguata sermia domain
    x_min = -263230.
    x_max = 130000.
    y_min = -2600000
    y_max = -2400000.
    
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
    high_log_speed = 2.0
    # log10 of min speed (m/yr) for cell spacing
    low_log_speed = 0.
    # distance at which cell spacing = max_spac (meters)
    high_dist = 1.e5
    # distance within which cell spacing = min_spac (meters)
    low_dist = 1.e4
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
    use_dist_to_edge = False
    use_bed = False

mesh_gen
--------

``landice/isunnguata_sermia/mesh_gen`` creates a variable resolution mesh.
The default is 1-10km resolution with mesh density determined by
observed ice speed. There is no model integration step.
