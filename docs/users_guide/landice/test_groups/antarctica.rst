.. _landice_antarctica:

antarctica
==========

The ``landice/antarctica`` test group includes a test case for creating a
mesh for Antarctica.

.. figure:: images/antarctica_8to80km.png
   :width: 777 px
   :align: center

   Ice thickness on Antarctica 8-80km variable resolution mesh.

config options
--------------

The test group uses the following default config options.  At this point only
the mesh generation options are adjusted through the config file.

.. code-block:: cfg

    [antarctica]

    # number of levels in the mesh
    levels = 5

    # distance from ice margin to cull (km).
    # Set to a value <= 0 if you do not want
    # to cull based on distance from margin.
    cull_distance = 70.0

    # mesh density parameters
    # minimum cell spacing (meters)
    min_spac = 8.e3
    # maximum cell spacing (meters)
    max_spac = 8.e4
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
    use_dist_to_grounding_line = True
    use_dist_to_edge = False

mesh_gen
--------

``landice/antarctica/mesh_gen`` creates a 8-80km variable resolution mesh.
There is no model integration step.
