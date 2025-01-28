.. _dev_ocean_tides:

tides
=========

The ``tides`` test group implements single layer
barotropic, tidal cases as described in :ref:`ocean_tides` in
the User's Guide.

initial_state
-------------
The class :py:class:`compass.ocean.tests.tides.init.initial_state.InitialState`
defines a step for running MPAS-Ocean in init mode. The vertical mesh is
set up with a single layer. 

interpolate_wave_drag
---------------------
The class :py:class:`compass.ocean.tests.tides.init.interpolate_wave_drag.InterpolateWaveDrag`
defines a step for interpolating HYCOM data onto the MPAS-O mesh
for the topographic wave drag scheme.

remap_bathymetry
----------------
The class :py:class:`compass.ocean.tests.tides.init.remap_bathymetry.RemapBathymetry`
defines a step to perform an integral remap of bathyetry data onto the MPAS-O mesh.

forward
-------
The class :py:class:`compass.ocean.tests.tides.forward.forward.ForwardStep`
defines a step to run MPAS-Ocean in forward mode.

analysis
--------
The class :py:class:`compass.ocean.tests.tides.analysis.Analysis`
defines a step to extract harmonic constituent data from the TPXO database.
These values are used to compute and plot errors.


