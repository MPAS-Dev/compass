.. _dev_landice_issunguata_sermia:

issunguata_sermia
=============

The ``issunguata_sermia`` test group (:py:class:`compass.landice.tests.issunguata_sermia.IssunguataSermia`)
creates a variable resolution (default 1-10 km) mesh for a regional domain of Issunguata Sermia, Greenland
(see :ref:`landice_issunguata_sermia`).

.. _dev_landice_issunguata_sermia_framework:

framework
---------

The shared config options for the ``issunguata_sermia`` test group are described
in :ref:`landice_issunguata_sermia` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.issunguata_sermia.mesh.Mesh`
defines a step for creating a variable resolution Issunguata Sermia mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.issunguata_sermia.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.issunguata_sermia.mesh.Mesh` to create
the variable resolution Issunguata Sermia mesh.
