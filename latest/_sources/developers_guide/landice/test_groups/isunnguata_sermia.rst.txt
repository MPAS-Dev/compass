.. _dev_landice_isunnguata_sermia:

isunnguata_sermia
=================

The ``isunnguata_sermia`` test group (:py:class:`compass.landice.tests.isunnguata_sermia.IsunnguataSermia`)
creates a variable resolution (default 1-10 km) mesh for a regional domain of Isunnguata Sermia, Greenland
(see :ref:`landice_isunnguata_sermia`).

.. _dev_landice_isunnguata_sermia_framework:

framework
---------

The shared config options for the ``isunnguata_sermia`` test group are described
in :ref:`landice_isunnguata_sermia` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.isunnguata_sermia.mesh.Mesh`
defines a step for creating a variable resolution Isunnguata Sermia mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.isunnguata_sermia.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.isunnguata_sermia.mesh.Mesh` to create
the variable resolution Isunnguata Sermia mesh.
