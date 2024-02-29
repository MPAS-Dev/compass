.. _dev_landice_antarctica:

antarctica
==========

The ``antarctica`` test group (:py:class:`compass.landice.tests.antarctica.Antarctica`)
creates a variable resolution 4-20 km mesh for the whole region of Antarctica.
(see :ref:`landice_antarctica`). Here, we describe framework for this test group.

.. _dev_landice_antarctica_framework:

framework
---------

The config options for the ``antarctica`` test group are described
in :ref:`landice_antarctica` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.antarctica.mesh.Mesh`
defines a step for creating a variable resolution Antarctica mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.antarctica.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.antarctica.mesh.Mesh` to create
the variable resolution Antarctica mesh.  It makes extensive use of
functions in :py:class:`compass.landice.mesh`.

The mesh generation is based around an 8 km reference dataset, which
is updated in a number of ways directly from high resolution source
data.  In the future, a complete workflow from source datasets may
replace this hybrid method.

First, ``add_bedmachine_thk_to_ais_gridded_data`` is used to update
the thickness field in the reference dataset with that from BedMachine
for the purposes of defining an ice-sheet extent and mesh resolution
(not the actual thickness interpolation).

Second, ``build_cell_width`` is called in the standard way to define the
resolution density function.

Third, ``preprocess_ais_data`` is called to perform a series of adjustments
to the AIS datasets needed to work with the rest of the workflow.

From here, dataset interpolation happens, starting with the standard
interpolation script applied to the standard dataset, followed by bespoke
interpolation of the high resolution BedMachine and MEASURES datasets.
The step completes with some clean up and creation of a graph file.
