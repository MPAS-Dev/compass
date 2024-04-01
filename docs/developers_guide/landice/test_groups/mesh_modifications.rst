.. _dev_mesh_modifications:

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

* create a cull mask, either from a region mask with specified region number
  or from a geojson file

* optionally extend culling mask a certain number of cells into the ocean
  along the ice-shelf front if using a region mask

* cull mesh, convert mesh, mark horns for culling, cull again

* set lat/lon fields in mesh based on specified projection

* interpolate fields from source mesh to subdomain mesh using
  nearest neighbor interpolation.
  This can be done either with the ``interpolate_to_mpasli_grid.py`` script
  or using ``ncremap``.  The ncremap method is slower and potentially more
  fragile (depends on more external tools), but if used allows the option
  to remap additional files (e.g. forcing files).

* mark domain boundary cells as Dirichlet velocity conditions

* create a graph file for the subdomain mesh

* optionally remap additional files that use the same source mesh

The method using ncremap uses pyremap to create the mapping file
and then calls ncremap with some pre and post processing steps.
Those operations happen in a helper function nameed ``_remap_with_ncremap``.
