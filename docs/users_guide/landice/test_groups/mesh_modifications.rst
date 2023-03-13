.. _mesh_modifications:

mesh_modifications
==================

The ``mesh_modifications`` test group includes test cases for modifying
existing meshes.

It currently contains one test case for extracting a subdomain from an
existing larger domain.

subdomain_extractor
-------------------

``landice/mesh_modifications/subdomain_extractor`` extracts a subdomain from a
larger domain.  The extraction is defined by a specified region in a
regionCellMask file.  In the future, this test could be extended to optionally
use a GeoJSON file for defining the culling mask instead.
The user should modify the default config for their application.

In the future, the ability to apply the extractor to forcing files as well may
be added.

config options
~~~~~~~~~~~~~~

The ``subdomain_extractor`` test case uses the following default config
options.  They should be adjusted by the user before setting up and running
the test case.

.. code-block:: cfg

    [subdomain]

    # path to file from which to extract subdomain
    source_file = TO BE SUPPLIED BY USER

    # path to region mask file for source_file
    region_mask_file = TO BE SUPPLIED BY USER

    # region number to extract
    region_number = 1

    # filename for the subdomain to be generated
    dest_file_name = subdomain.nc

    # mesh projection to be used for setting lat/lon values
    # Should match the projection used in the source_file
    # Likely one of 'ais-bedmap2' or 'gis-gimp'
    mesh_projection = ais-bedmap2

    # whether to extend mesh into the open ocean along the ice-shelf margin
    # This is necessary if the region mask ends right along the ice-shelf edge,
    # or if the ice-shelf is covered by two regions.
    # It is recommended to try extracting a subdomain with this False, and if the
    # ocean buffer is inadequate, trying again with this True.
    extend_ocean_buffer = False

    # number of iterations to grow open ocean buffer along ice-shelf margin
    # Only used if extend_ocean_buffer is True
    # Should be equal to approximately the number of ocean buffer cells in the
    # source_file
    grow_iters = 15
