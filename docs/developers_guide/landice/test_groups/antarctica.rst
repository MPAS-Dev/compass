.. _dev_landice_antarctica:

antarctica
==========

The ``antarctica`` test group (:py:class:`compass.landice.tests.antarctica.Antarctica`)
creates a variable resolution 8-80 km mesh for the whole region of Antarctica.
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
the 8-80 km variable resolution Antarctica mesh.