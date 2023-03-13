.. _mesh_modifications:

mesh_modifications
==================

The ``mesh_modifications`` test group includes test cases for modifying
existing meshes.
(see :ref:`mesh_modifications`).

It currently contains one test case for extracting a subdomain from an
existing larger domain.

framework
---------

There is no shared framework for this test group.

subdomain_extractor
-------------------

The class :py:class:`compass.landice.tests.mesh_modifications.subdomain_extractor.SubdomainExtractor`
extracts a subdomain from a larger domain.  It simply calls the class
:py:class:`compass.landice.tests.mesh_modifications.subdomain_extractor.extract_region.ExtractRegion`.

extract_region
--------------

The :py:class:`compass.landice.tests.mesh_modifications.subdomain_extractor.extract_region.ExtractRegion`
class performs the operations to extract a region subdomain from a larger
domain.  Using user-supplied config information, it performs the following
steps:

* create a cull mask

* optionally extend culling mask a certain number of cells into the ocean
  along the ice-shelf front

* cull mesh, convert mesh, mark horns for culling, cull again

* create a landice mesh from base MPAS mesh

* set lat/lon fields in mesh based on specified projection

* interpolate data fields from source mesh to subdomain mesh

* mark domain boundary cells as Dirichlet velocity conditions

* create a graph file for the subdomain mesh
