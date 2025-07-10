.. _dev_landice_humboldt:

humboldt
========

The ``humboldt`` test group (:py:class:`compass.landice.tests.humboldt.Humboldt`)
creates a variable resolution 1-10 km mesh for a regional domain of Humboldt Glacier
(see :ref:`landice_humboldt`), and includes a number of tests running MALI on a
pre-generated mesh.  Here, we describe the shared framework for this test group.

.. _dev_landice_humboldt_framework:

framework
---------

The shared config options for the ``humboldt`` test group are described
in :ref:`landice_humboldt` in the User's Guide.

mesh
~~~~

The class :py:class:`compass.landice.tests.humboldt.mesh.Mesh`
defines a step for creating a variable resolution Humboldt Glacier mesh.
This is used by the ``mesh_gen`` test case.

mesh_gen
--------

The :py:class:`compass.landice.tests.humboldt.mesh_gen.MeshGen`
calls the :py:class:`compass.landice.tests.humboldt.mesh.Mesh` to create
the 1-10 km variable resolution Humboldt Glacier mesh.

run_model
---------
The :py:class:`compass.landice.tests.humboldt.run_model.RunModel` defines
the process for setting up and running a MALI simulation for the humboldt
configuration.  It is called by
:py:class:`compass.landice.tests.humboldt.decomposition_test.DecompositionTest`
and
:py:class:`compass.landice.tests.humboldt.restartn_test.RestartTest`.

decomposition_test
------------------
The compass.landice.tests.humboldt.decomposition_test.DecompositionTest
performs the same simulation on different numbers of cores. It ensures
relevant variables are identical or have expected differences.

restart_test
------------
The compass.landice.tests.humboldt.restart_test.RestartTest performs a
run of a full specified duration followed by a short run plus a restart
to equal the same total duration.  It checks that relevant variables
are bit-for-bit when doing a restart.

The restart step works by creating two different namelist and streams files,
one each with landice as the suffix and one each with landice.rst as the
suffix.




