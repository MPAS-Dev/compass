.. _landice_humboldt:

humboldt
========

The ``landice/humboldt`` test group creates a 1-10km Humboldt Glacier mesh. 

.. figure:: images/humboldt_1to10km.png
   :width: 777 px
   :align: center

   Ice thickness on Humboldt 1-10km variable resolution mesh.

The test group includes a single test case that creates the variable resolution mesh.

config options
--------------

The test group uses the following default config options:

.. code-block:: cfg

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

default
-------

``landice/humboldt/default`` createst the 1-10km variable resolution mesh. 
There is no model integration step.
