.. _dev_landice_humboldt:

humboldt
========

The ``humboldt`` test group (:py:class:`compass.landice.tests.humboldt.Humboldt`)
creates a variable resolution 1-10 km mesh for a regional domain of Humboldt Glacier
(see :ref:`landice_humboldt`).  Here, we describe the shared framework for
this test group and the singe test case.

.. _dev_landice_humboldt_framework:

framework
---------

The shared config options for the ``humboldt`` test group are described
in :ref:`landice_humboldt` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.humboldt.mesh.Mesh`
defines a step for creating a variable resolution Humboldt Glacier mesh.
This is used by the ``default`` test case.

default
-------

The :py:class:`compass.landice.tests.humboldt.default.Default`
calls the :py:class:`compass.landice.tests.humboldt.mesh.Mesh` to create
the 1-10 km variable resolution Humboldt Glacier mesh.


