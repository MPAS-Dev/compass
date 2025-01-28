.. _landice_koge_bugt_s:

koge_bugt_s
===========

The ``landice/koge_bugt_s`` test group includes a test case for creating a
mesh for Koge Bugt S, Greenland. The optimization for basal friction
happens outside of COMPASS because it requires expert usage and takes a
larger amount of computing resources than COMPASS is typically run with.

.. figure:: images/koge_bugt_s_500m_to_4km.png
   :width: 777 px
   :align: center

   Ice thickness and grid spacing in meters on Koge Bugt S 500m-4km variable resolution mesh.

The test group includes a single test case that creates the variable resolution mesh.

config options
--------------

The test group uses the following default config options.  At this point only
the mesh generation options are adjusted through the config file.

.. code-block:: cfg

    [mesh]

    # number of levels in the mesh
    levels = 10

    # distance from ice margin to cull (km).
    # Set to a value <= 0 if you do not want
    # to cull based on distance from margin.
    cull_distance = 5.0

    # mesh density parameters
    # minimum cell spacing (meters)
    min_spac = 5.e2
    # maximum cell spacing (meters)
    max_spac = 4.e3
    # log10 of max speed (m/yr) for cell spacing
    high_log_speed = 2.5
    # log10 of min speed (m/yr) for cell spacing
    low_log_speed = 0.75
    # distance at which cell spacing = max_spac (meters)
    high_dist = 1.e5
    # distance within which cell spacing = min_spac (meters)
    low_dist = 1.e4

    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = False
    use_dist_to_edge = True

mesh_gen
--------

``landice/koge_bugt_s/default`` creates a variable resolution mesh.
The default is 500m-4km resolution with mesh density determined by
observed ice speed and distance to ice margin. There is no model
integration step.
