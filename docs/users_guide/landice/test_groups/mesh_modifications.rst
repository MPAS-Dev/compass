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
larger domain.  The region to be extracted can be defined be either a region_mask
file with a region number or with a geojson file.

If using a region_mask file, there is the option of extending the mask into the open
ocean by a specified number of cells.  This was implemented because many region
masks end at the present-day ice front, which would leave no gutter on the
subdomain mesh.  It is recommended to first try the subdomain extraction without
the ocean extension, and if results are clipped too close to the ice front, then
try adding an ocean buffer.  If results with the ocean buffer do not achieve
the desired effect, one should switch to the geojson method and manually create
or modify the region, e.g. in QGIS.

There are two methods available for performing the remapping: using ncremap
or standard MALI interpolation methods.  The standard MALI interpolation method
is faster and potentially less fragile, as it relies on fewer underlying tools.
However, the ncremap option provides the ability to additionally remap up to five
ancillary files (e.g. SMB, TF, basal melt param, von Mises parameter files).
The MALI interpolation method can likely be run quickly on a login node, but the
ncremap method will likely need a compute node.

The user should modify the default config for their application.

config options
~~~~~~~~~~~~~~

The ``subdomain_extractor`` test case uses the following default config
options.  They should be adjusted by the user before setting up and running
the test case.

.. code-block:: cfg

    [subdomain]

    # path to file from which to extract subdomain
    source_file = TO BE SUPPLIED BY USER

    # method for defining region
    # one of 'region_mask_file' or 'geojson'
    region_definition = region_mask_file

    # path to geojson file to be used if region_definition='geojson'
    geojson_file = TO BE SUPPLIED BY USER

    # path to region mask file for if region_definition='region_mask_file'
    region_mask_file = TO BE SUPPLIED BY USER

    # region number to extract if region_definition='region_mask_file'
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

    # method for performing interpolation
    # 'ncremap' uses pyremap to call ESMF_RegridWeightGen to generate a
    # nstd weight file and then uses ncremap to perform remapping.
    # This method supports interpolating ancillary files (below)
    # but likely needs to be run on a compute node and is more fragile.
    # 'mali_interp' uses the MALI interpolation script interpolate_to_mpasli_grid.py
    # This method does not support ancillary files but may be more robust
    # and can likely be run quickly on a login node.
    interp_method = ncremap

    # optional forcing files that could also be interpolated
    # e.g. SMB or TF files
    # interpolating these files requires using the 'ncremap' interp_method
    extra_file1 = None
    extra_file2 = None
    extra_file3 = None
    extra_file4 = None
    extra_file5 = None
