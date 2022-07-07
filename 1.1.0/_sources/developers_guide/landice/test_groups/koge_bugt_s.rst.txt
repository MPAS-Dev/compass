.. _dev_landice_koge_bugt_s:

koge_bugt_s
===========

The ``koge_bugt_s`` test group (:py:class:`compass.landice.tests.koge_bugt_s.KogeBugtS`)
creates a variable resolution (default 500m to 4km) mesh for a regional domain of Koge Bugt S, Greenland
(see :ref:`landice_koge_bugt_s`).

.. _dev_landice_koge_bugt_s_framework:

framework
---------

The shared config options for the ``koge_bugt_s`` test group are described
in :ref:`landice_koge_bugt_s` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.koge_bugt_s.mesh.Mesh`
defines a step for creating a variable resolution Koge Bugt S mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.koge_bugt_s.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.koge_bugt_s.mesh.Mesh` to create
the variable resolution Koge Bugt S mesh.
