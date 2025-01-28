.. _landice_thwaites:

thwaites
=========

The ``landice/thwaites`` test group runs tests with a coarse (4-14-km)
`Thwaites Glacier mesh <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/thwaites.4km.210608.nc>`_.
The purpose of this test group is to provide a realistic glacier that
includess an ice shelf.
The mesh and initial condition are already generated.  In the future,
additional test cases may be added for generating a new version of the
Thwaites mesh at different resolutions and using different data sources.

.. figure:: images/thwaites_speed.png
   :width: 777 px
   :align: center

   FO velocity solution visualized in Paraview.  Grounding line position
   shown with a white line.

The test group includes three test cases, two of which have one or more steps
that are variants on ``run_model`` (given other names in the decomposition and
restart test cases to distinguish multiple model runs), which performs time
integration of the model.  There is a not an explicit smoke test, but the
``full_run`` step of the ``restart_test`` can be used as a smoke test.

The ``decomposition_test`` and ``restart_test`` test cases in this test group
can only be run with the FO velocity solvers. Running with the FO solver requires
a build of MALI that includes Albany. There is no integration step for the test
case ``mesh_gen``.

config options
--------------

The ``mesh_gen`` test case uses the default config options below.
The other test cases do not use config options.

.. code-block:: cfg

    [mesh]

    # number of levels in the mesh
    levels = 10

    # distance from ice margin to cull (km).
    # Set to a value <= 0 if you do not want
    # to cull based on distance from margin.
    cull_distance = 10.0
    
    # mesh density parameters
    # minimum cell spacing (meters)
    min_spac = 1.e3
    # maximum cell spacing (meters)
    max_spac = 8.e3
    # log10 of max speed for cell spacing
    high_log_speed = 2.5
    # log10 of min speed for cell spacing
    low_log_speed = 0.75
    # distance at which cell spacing = max_spac
    high_dist = 1.e5
    # distance within which cell spacing = min_spac
    low_dist = 5.e4
    
    # mesh density functions
    use_speed = True
    use_dist_to_grounding_line = True
    use_dist_to_edge = True

decomposition_test
------------------

``landice/thwaites/decomposition_test`` runs short (2-day) integrations of the
model forward in time on 16 (``16proc_run`` step) and then on 32 cores
(``32proc_run`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs.

restart_test
------------

``landice/thwaites/restart_test`` first runs a short (5-day) integration
of the model forward in time (``full_run`` step).  Then, a second step
(``restart_run``) performs two subsequent 2 and 3 day integrations, where the
second begins from a restart file saved by the first. Prognostic variables
are compared between the "full" and "restart" runs to make sure they are
bit-for-bit identical.

mesh_gen
-------------

``landice/thwaites/mesh_gen`` creates a variable resolution mesh based
on the the config options listed above. This will not be the same as the
pre-generated 4-14km mesh used in ``decomposition_test`` and ``restart_test``
because it uses a newer version of Jigsaw. Note that the basal friction
optimization is performed separately and is not part of this test case.
