.. _dev_ocean_hurricane:

hurricane
=========

The ``hurricane`` test group implements regionally refined, single layer
barotropic, tropical cyclone cases as described in :ref:`ocean_hurricane` in
the User's Guide.

mesh
----
The class
:py:class:`compass.ocean.tests.hurricane.mesh.dequ120at30cr10rr2.DEQU120at30cr10rr2Mesh`
defines a step for generating the global regionally refined mesh for the test group. 
It inherits from the :py:class:`compass.ocean.tests.global_ocean.mesh.mesh.MeshStep`
and provides the :py:func:`compass.ocean.tests.hurricane.mesh.dequ120at30cr10rr2.DEQU120at30cr10rr2Mesh.build_cell_width_lat_lon()`
method to specify the mesh resolution function using the :py:func:`mpas_tools.ocean.coastal_tools.coastal_refined_mesh()`
function.

initial_state
-------------
The class :py:class:`compass.ocean.tests.hurricane.init.initial_state.InitialState`
defines a step for running MPAS-Ocean in init mode. The vertical mesh is
set up with a single layer. 


interpolate_atm_forcing
-----------------------
The class :py:class:`compass.ocean.tests.hurricane.init.interpolate_atm_forcing.InterpolateAtmForcing`
defines a step for interpolating CFSv2 reanalysis data for atmospheric winds
and pressure onto the MPAS-Ocean mesh at hourly time intervals. The forward
run uses this as input to update the time varying atmospheric forcing.


create_pointstats_file
----------------------
The class :py:class:`compass.ocean.tests.hurricane.init.create_pointstats_file.CreatePointstatsFile`
defines a step to create the input file for the MPAS-Ocean pointWiseStats
analysis member based on station locations which have observed data.

forward
-------
The class :py:class:`compass.ocean.tests.hurricane.forward.forward.ForwardStep`
defines a step to run MPAS-Ocean in forward mode

analysis
--------
The class :py:class:`compass.ocean.tests.hurricane.analysis.Analysis`
defines a step to generate validation plots comparing sea surface height
timeseries between modeled and observed data at several different stations.
Both NOAA and USGS observations are plotted.


