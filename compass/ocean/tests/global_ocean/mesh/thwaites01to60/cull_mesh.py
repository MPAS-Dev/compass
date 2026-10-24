import os
import xarray as xr
import numpy as np
from geometric_features import read_feature_collection
from mpas_tools.mesh.creation.signed_distance import mask_from_geojson

from compass.ocean.mesh.cull import CullMeshStep


class ThwaitesCullMeshStep(CullMeshStep):
    """
    Custom cull mesh step for Thwaites mesh with regional domain support

    This subclass extends the standard CullMeshStep to add regional culling.
    It creates a hook file that the modified cull.py will execute to modify
    the land mask after creation but before culling.
    """

    def setup(self):
        """
        Set up the step by adding input and output files
        """
        # Call parent setup
        super().setup()

        # Create hook file for land mask modification
        self._create_hook_file()

    def _create_hook_file(self):
        """
        Create the modify_land_mask_hook.py file that will be called
        during the culling process
        """
        hook_content = '''"""
Hook for modifying land mask before culling - auto-generated
"""
import os
import xarray as xr
import numpy as np
from geometric_features import read_feature_collection
from mpas_tools.mesh.creation.signed_distance import mask_from_geojson


def modify_land_mask(logger):
    """
    Modify land_mask.nc to add regional domain boundaries

    This is called from compass.ocean.mesh.cull._cull_mesh_with_logging
    after land_mask.nc is created but before culling begins.
    """
    # Read config - need to get it from the step's config
    # This hook runs in the cull_mesh working directory
    from compass.config import CompassConfigParser
    config = CompassConfigParser()
    config.read('../compass.cfg')  # Read from test case level

    # Check config section
    if not config.has_section('thwaites01to60'):
        logger.info('No thwaites01to60 config section')
        return

    section = config['thwaites01to60']

    # Check if regional domain is specified
    has_geojson = section.has_option('regional_domain_geojson')
    has_bounds = (section.has_option('lat_min') and
                  section.has_option('lon_min'))

    if not (has_geojson or has_bounds):
        logger.info('No regional domain specified - using standard land mask')
        return

    logger.info('====================================================')
    logger.info('APPLYING REGIONAL MASK MODIFICATION FOR THWAITES')
    logger.info('====================================================')

    # Load base mesh to get cell coordinates
    ds_mesh = xr.open_dataset('base_mesh.nc')

    # Load land mask
    ds_mask = xr.open_dataset('land_mask.nc')

    # Find the mask variable
    if 'regionCellMasks' in ds_mask:
        mask_var = 'regionCellMasks'
    elif 'landIceMask' in ds_mask:
        mask_var = 'landIceMask'
    else:
        mask_vars = [v for v in ds_mask.variables if 'mask' in v.lower()]
        if mask_vars:
            mask_var = mask_vars[0]
            logger.warning(f'Using mask variable: {mask_var}')
        else:
            logger.error('No mask variable found in land_mask.nc')
            return

    land_mask = ds_mask[mask_var]

    # Create mask for cells INSIDE the regional domain
    if has_geojson:
        geojson_file = section.get('regional_domain_geojson')
        logger.info(f'  Using domain from {geojson_file}')
        fc = read_feature_collection(geojson_file)
        inside_region = mask_from_geojson(
            fc,
            np.degrees(ds_mesh.lonCell.values),
            np.degrees(ds_mesh.latCell.values))
    else:
        # Use lat-lon bounds
        lat_min = section.getfloat('lat_min')
        lat_max = section.getfloat('lat_max')
        lon_min = section.getfloat('lon_min')
        lon_max = section.getfloat('lon_max')

        logger.info(f'  Using domain bounds:')
        logger.info(f'    lat: [{lat_min}, {lat_max}]')
        logger.info(f'    lon: [{lon_min}, {lon_max}]')

        lat_deg = np.degrees(ds_mesh.latCell.values)
        lon_deg = np.degrees(ds_mesh.lonCell.values)

        # Handle longitude wrapping
        lon_deg = np.where(lon_deg > 180, lon_deg - 360, lon_deg)

        inside_region = ((lat_deg >= lat_min) & (lat_deg <= lat_max) &
                        (lon_deg >= lon_min) & (lon_deg <= lon_max))

    # Cells OUTSIDE the region get added to land mask
    outside_region = ~inside_region

    # Count cells
    ncells_total = len(inside_region)
    ncells_outside = int(outside_region.sum())
    ncells_inside = int(inside_region.sum())
    ncells_originally_land = int((land_mask.values > 0).sum())

    logger.info(f'  Mesh statistics:')
    logger.info(f'    Total cells: {ncells_total}')
    logger.info(f'    Cells inside region: {ncells_inside}')
    logger.info(f'    Cells outside region: {ncells_outside}')
    logger.info(f'    Originally land: {ncells_originally_land}')

    # Modify land mask
    if np.issubdtype(land_mask.dtype, np.integer):
        modified_mask = np.where(outside_region, 1, land_mask.values)
    else:
        modified_mask = np.logical_or(outside_region, land_mask.values > 0)

    ncells_new_land = int((modified_mask > 0).sum())
    logger.info(f'    After modification: {ncells_new_land} land cells')
    logger.info(f'    Added {ncells_new_land - ncells_originally_land} cells to land mask')

    # Backup original
    if os.path.exists('land_mask.nc'):
        os.rename('land_mask.nc', 'land_mask_original.nc')

    # Update dataset
    ds_mask[mask_var] = (land_mask.dims, modified_mask)
    ds_mask[mask_var].attrs.update(land_mask.attrs)
    ds_mask[mask_var].attrs['comment'] = (
        'Land mask modified to include regional domain boundaries for Thwaites mesh')

    # Write modified mask
    ds_mask.to_netcdf('land_mask.nc')

    logger.info('  Modified land mask written successfully')
    logger.info('  (Original backed up to land_mask_original.nc)')
    logger.info('====================================================')
'''

        # Write hook file to the step's path
        # This will be in the cull_mesh subdirectory when the step runs
        hook_file = os.path.join(self.path, 'modify_land_mask_hook.py')
        os.makedirs(os.path.dirname(hook_file), exist_ok=True)
        with open(hook_file, 'w') as f:
            f.write(hook_content)
