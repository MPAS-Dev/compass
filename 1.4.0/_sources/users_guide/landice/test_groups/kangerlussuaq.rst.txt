.. _landice_kangerlussuaq:

kangerlussuaq
=============

The ``landice/kangerlussuaq`` test group includes a test case for creating a
mesh for Kangerlussuaq Glacier, Greenland. The optimization for basal friction
happens outside of COMPASS because it requires expert usage and takes a
larger amount of computing resources than COMPASS is typically run with.

.. figure:: images/kangerlussuaq_1to10km.png
   :width: 777 px
   :align: center

   Ice thickness and grid spacing in meters on Kangerlussuaq 1-10km variable resolution mesh.

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

    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = False
    use_dist_to_edge = True

mesh_gen
--------

``landice/kangerlussuaq/default`` creates a variable resolution mesh.
The default is 1-10km resolution with mesh density determined by
observed ice speed and distance to ice margin. There is no model
integration step.
