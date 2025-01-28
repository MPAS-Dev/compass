.. _dev_landice_crane:

crane
=============

The ``crane`` test group (:py:class:`compass.landice.tests.crane.Crane`)
creates a variable resolution (default 500 m - 1 km) mesh for a regional domain of Crane Glacier
(see :ref:`landice_crane`).

.. _dev_landice_crane_framework:

framework
---------

The shared config options for the ``crane`` test group are described
in :ref:`landice_crane` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.crane.mesh.Mesh`
defines a step for creating a variable resolution Crane Glacier mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.crane.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.crane.mesh.Mesh` to create
the variable resolution Crane Glacier mesh.
