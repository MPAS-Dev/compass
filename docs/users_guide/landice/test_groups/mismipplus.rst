.. _landice_mismipplus:

mismipplus
==========

The ``mismipplus`` test group (:py:class:`compass/landice/tests/mismipplus`)
implements a limited number of the expeirments from the Marine Ice Sheet 
Model Intercomparison Project Plus (MISMIP+; 
`Asay-Davis et al., 2016 <https://gmd.copernicus.org/articles/9/2471/2016/gmd-9-2471-2016.pdf>`_).
To date, the spinup and ``Ice0`` are the only expeirments currently supported. 
In the future, additional test cases may be added for the missing MISMIP+ 
experiments (i.e. ``Ice1r``, ``Ice1ra``, ``Ice1rr``, ``Ice2r``, ``Ice2r``, ``Ice2ra``, 
``Ice2rr``; Table 2. `Asay-Davis et al., 2016 <https://gmd.copernicus.org/articles/9/2471/2016/gmd-9-2471-2016.pdf>`_).

shared config options 
---------------------

Both ``mismipplus`` test cases share the following config options:

.. code-block:: cfg

    [mismipplus]
    
    # the number of cells per core to aim for
    goal_cells_per_core = 300
    
    # the approximate maximum number of cells per core
    max_cells_per_core = 5000
    
    # number of cells at 1000m resolution (with no gutter present). This is 
    # used as huerisitc to scale the number of cells with resolution, in 
    # order to constrain the resoucres (i.e. `ntasks`) based on the mesh size.
    ncells_at_1km_res = 61382

The config options are all aimed at dynamically calculating the optimal number 
of ``ntasks`` for the ``run_model`` step based on the resolution requested 
by the test case. For more information about how ``ntasks`` is calculated see 
:ref:`dev_landice_mismipplus_tasks` in the developers guide page.

.. _landice_mismipplus_smoke_test:

smoke_test
----------

``landice/mismipplus/smoke_test`` runs the MISMIP+ ``Ice0`` experiment for 
five years using a spun-up 
`2km mesh <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/MISMIP_2km_20220502.nc>`_.
The simulation is run with the adaptive timestepper on. 

.. code-block:: cfg
   
    # config options for the smoke test
    [mesh]
    
    # Resolution (m) of the spun up testcase, in longtime archiving. 
    resolution = 2000.

Currently 2000m is the only resolution supported. In the future, the 
``spin_up`` test case will be used to generate new versions of the
mesh at 8000m, 4000m, 2000m, and 1000m resolutions. 

.. _landice_mismipplus_spin_up:

spin_up
-------

``landice/mismipplus/spin_up`` generates the files needed to run a 20,000 year
spin-up using the standard MISMIP+ initial conditions outline in 
`section 2.1 of (Asay-Davis et al., 2016) <https://gmd.copernicus.org/articles/9/2471/2016/gmd-9-2471-2016.pdf>`_.
The purpose of this test case is generate a semi-realistic glacier geometry 
that includes an ice shelf, which can be used to the subsequent MISMIP+ 
experiments and/or additional experiments that require 3-D complexity without 
site-specific complications arising from using an actual glacier geometry. 


.. code-block:: cfg

    # config options for the mesh (i.e. initial condition) setup
    [mesh]
    
    # nominal edge length (m). resolution is "nominal" since the true edge 
    # length will be interactively determined such that the cell center to 
    # cell center distance in the y direction is exactly the y domain 
    # length (80 km).
    resolution = 8000.
    
    # length (m) to extend the eastern domain boundary by. Will be needed for 
    # simulations that use a dynamic calving law, where the claving front will
    # be irregularly shaped. Any value less than `2*resolution` will ignored,
    # as the default gutter length is 2 gridcells.
    gutter_length = 0.
    
    # ice density (kg m^{-3}). MISMIP+ uses 918 
    # (Table 1. Asay-Davis et al. 2016), but MALI defaults to 910. 
    # The user can choose if they want to stricly follow MISMIP+ or use 
    # the default MALI value.
    ice_density = 918.
    
    # Number of vertical levels
    levels = 10
    
    # Initial ice thickness (m)
    init_thickness = 100.
    
    # How to distribute vertical layers. Options are "glimmer" or "uniform".
    # "glimmer" will distribute the layer non-uniformily following
    # Eqn. (15) from Rutt et al. 2009.
    vetical_layer_distribution = glimmer 

