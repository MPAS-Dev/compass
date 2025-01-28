.. _dev_landice_kangerlussuaq:

kangerlussuaq
=============

The ``kangerlussuaq`` test group (:py:class:`compass.landice.tests.kangerlussuaq.Kangerlussuaq`)
creates a variable resolution (default 1-10 km) mesh for a regional domain of Kangerlussuaq Glacier
(see :ref:`landice_kangerlussuaq`).

.. _dev_landice_kangerlussuaq_framework:

framework
---------

The shared config options for the ``kangerlussuaq`` test group are described
in :ref:`landice_kangerlussuaq` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.kangerlussuaq.mesh.Mesh`
defines a step for creating a variable resolution Kangerlussuaq Glacier mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.kangerlussuaq.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.kangerlussuaq.mesh.Mesh` to create
the variable resolution Kangerlussuaq Glacier mesh.
