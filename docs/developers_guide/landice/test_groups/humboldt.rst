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

The class :py:class:`compass.landice.tests.thwaites.mesh.Mesh`
defines a step for creating a variable resolution Humboldt Glacier mesh.
This is used by the ``default`` test case.

.. _dev_landice_thwaites_decomposition_test:

default
-------

The :py:class:`compass.landice.tests.thwaites.default.Default`
calls the :py:class:`compass.landice.tests.thwaites.mesh.Mesh` to create
the 1-10 km variable resolution Humboldt Glacier mesh.


