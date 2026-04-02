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

Optional BedMachine and MEaSUREs interpolation can be enabled through
``[mesh]`` config options ``nProcs``, ``src_proj``, ``data_path``,
``bedmachine_filename``, and ``measures_filename``. If enabled, source
datasets are subset to the
configured mesh bounding box before SCRIP generation and conservative
remapping.
The ``src_proj`` option is used for optional remapping only; the
base-mesh projection in ``build_mali_mesh()`` is fixed for this test case.
