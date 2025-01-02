.. _dev_ocean_tides:

tides
=====

The ``tides`` test group implements single layer
barotropic, tidal cases as described in :ref:`ocean_tides` in
the User's Guide.

mesh test case
--------------
This test case generates the bathymetric dataset and horizontal mesh for tidal simulations.
A :py:class:`compass.ocean.tests.tides.mesh.Mesh` object is created with ``mesh_name`` as one of its
arguments. Based on this argument, it determines the appropriate child class of 
:py:class:`compass.mesh.spherical.SphericalBaseStep` to create the base mesh and adds a
:py:class:`compass.ocean.mesh.cull.CullMeshStep`. Prior to creation of the base mesh, a
:py:class:`compass.ocean.tests.tides.dem.CreatePixelFile` step is added to create the 
"pixel" files used to remap bathymetry data onto the mesh in the initial state test case.

meshes
~~~~~~
``tides`` currently defines 2 meshes, with more to come.

Icos7
^^^^^
This is a uniform resolution mesh based on icosahedral subdivision. It has approximately 60 km
resolution globally. It is defined by :py:class:`compass.mesh.spherical.IcosahedralMeshStep`.

vr45to5
^^^^^^^
This is a variable resolution mesh which has refinement based on bathymetric depth and slope
criteria. It has a maximum resolution of 45 km and a minimum resolution of 5 km along coastlines
and steep bathymetric gradients. It is defined by :py:class:`compass.ocean.tests.tides.mesh.vr45to5.VRTidesMesh`
which inherits from :py:class:`compass.mesh.spherical.QuasiUniformSphericalMeshStep` and overrides
the ``build_cell_width_lat_lon`` method with the resolution specification previously described.


initial state test case
-----------------------
This test case contains steps to calculate parameters necessary for the 
wave drag schemes in MPAS-Ocean, remaps the bathymetry onto the mesh, and
generates the initial state. These steps are added in the construction of 
the :py:class:`compass.ocean.tests.tides.init.Init` object. 

initial_state
~~~~~~~~~~~~~
The class :py:class:`compass.ocean.tests.tides.init.initial_state.InitialState`
defines a step for running MPAS-Ocean in init mode. The vertical mesh is
set up with a single layer. 

calculate_wave_drag
~~~~~~~~~~~~~~~~~~~
The class :py:class:`compass.ocean.tests.tides.init.calculate_wave_drag.CalculateWaveDrag`
defines a step for calculating bathymetric slopes and interpolating buoyancy frequency data onto
the MPAS-O mesh for the topographic wave drag parameterizations.

remap_bathymetry
~~~~~~~~~~~~~~~~
The class :py:class:`compass.ocean.tests.tides.init.remap_bathymetry.RemapBathymetry`
defines a step to perform an integral remap of bathymetry data onto the MPAS-O mesh.

forward test case
-----------------
The forward test case contains steps to run the forward simulation and 
compare the harmonic constituents with the TPXO database. These steps
are added in the creation of the :py:class:`compass.ocean.tests.tides.forward.Forward`
object.

forward
~~~~~~~
The class :py:class:`compass.ocean.tests.tides.forward.forward.ForwardStep`
defines a step to run MPAS-Ocean in forward mode.

analysis
~~~~~~~~
The class :py:class:`compass.ocean.tests.tides.analysis.Analysis`
defines a step to extract harmonic constituent data from the TPXO database.
These values are used to compute and plot errors.


