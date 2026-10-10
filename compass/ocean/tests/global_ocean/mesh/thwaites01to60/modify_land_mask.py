from compass.step import Step
import xarray as xr
from geometric_features import read_feature_collection
from mpas_tools.mesh.creation.signed_distance import mask_from_geojson
import numpy as np


class ModifyLandMask(Step):
    """
    Modify land mask to add regional domain boundaries

    This step runs BEFORE the standard cull_mesh step. If a regional domain
    is specified (via geojson or lat-lon bounds), cells outside the domain
    are marked as "land" in the mask. This causes them to be culled in the
    standard cull_mesh step that follows.

    If no regional domain is specified, this step passes the land mask
    through unchanged.
    """

    def __init__(self, test_case, base_mesh_step, name='modify_land_mask',
                 subdir=None):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        base_mesh_step : compass.Step
            The base mesh generation step

        name : str, optional
            The name of the step

        subdir : str, optional
            The subdirectory for the step
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

        self.base_mesh_step = base_mesh_step

    def setup(self):
        """
        Set up the step by adding input and output files
        """
        # We need the base mesh to get cell coordinates
        base_mesh_path = self.base_mesh_step.path
        base_mesh_filename = self.base_mesh_step.config.get(
            'spherical_mesh', 'mpas_mesh_filename')
        self.add_input_file(
            filename='base_mesh.nc',
            work_dir_target=f'{base_mesh_path}/{base_mesh_filename}')

        # Input: land_mask.nc from geometric_features processing
        # This should exist from the standard global_ocean workflow
        self.add_input_file(filename='land_mask.nc',
                           target='land_mask.nc')

        # Output: modified land mask (or same if no regional domain)
        self.add_output_file(filename='land_mask_with_region.nc')

    def run(self):
        """
        Run the step - modify land mask if regional domain is specified
        """
        config = self.config
        logger = self.logger

        # Check config section - try both specific and general sections
        if config.has_section('thwaites01to60'):
            section_name = 'thwaites01to60'
        elif config.has_section('spherical_mesh'):
            section_name = 'spherical_mesh'
        else:
            section_name = None

        # Check if regional domain is specified
        has_geojson = (section_name and
                      config.has_option(section_name, 'regional_domain_geojson'))
        has_bounds = (section_name and
                     config.has_option(section_name, 'lat_min') and
                     config.has_option(section_name, 'lon_min'))

        if not (has_geojson or has_bounds):
            # No regional domain - pass through unchanged
            logger.info('No regional domain specified - using standard land mask')
            import os
            os.symlink('land_mask.nc', 'land_mask_with_region.nc')
            return

        logger.info('Modifying land mask to add regional boundaries...')

        # Load base mesh to get cell coordinates
        ds_mesh = xr.open_dataset('base_mesh.nc')

        # Load land mask
        ds_mask = xr.open_dataset('land_mask.nc')

        # Determine which mask variable to use
        # Different global_ocean meshes may use different variable names
        if 'regionCellMasks' in ds_mask:
            mask_var = 'regionCellMasks'
        elif 'landIceMask' in ds_mask:
            mask_var = 'landIceMask'
        else:
            # Find any mask-like variable
            mask_vars = [v for v in ds_mask.variables if 'mask' in v.lower()]
            if mask_vars:
                mask_var = mask_vars[0]
                logger.warning(f'Using mask variable: {mask_var}')
            else:
                raise ValueError('No mask variable found in land_mask.nc')

        land_mask = ds_mask[mask_var]

        section = config[section_name]

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

            logger.info(f'  Using bounds: '
                       f'lat [{lat_min}, {lat_max}], '
                       f'lon [{lon_min}, {lon_max}]')

            lat_deg = np.degrees(ds_mesh.latCell.values)
            lon_deg = np.degrees(ds_mesh.lonCell.values)

            # Handle longitude wrapping (convert to -180 to 180 range)
            lon_deg = np.where(lon_deg > 180, lon_deg - 360, lon_deg)

            inside_region = ((lat_deg >= lat_min) & (lat_deg <= lat_max) &
                           (lon_deg >= lon_min) & (lon_deg <= lon_max))

        # Cells OUTSIDE the region get added to land mask
        # (i.e., they will be culled by the standard cull_mesh step)
        outside_region = ~inside_region

        # Count how many cells will be affected
        ncells_total = len(inside_region)
        ncells_outside = outside_region.sum()
        ncells_inside = inside_region.sum()
        ncells_already_land = (land_mask > 0).sum()

        logger.info(f'  Total cells: {ncells_total}')
        logger.info(f'  Cells inside region: {ncells_inside}')
        logger.info(f'  Cells outside region: {ncells_outside}')
        logger.info(f'  Cells already land: {ncells_already_land}')

        # Modify land mask: original land + cells outside region
        # Use logical OR to combine masks
        if np.issubdtype(land_mask.dtype, np.integer):
            # Integer mask: 1 = land, 0 = ocean
            modified_mask = np.where(outside_region, 1, land_mask.values)
        else:
            # Boolean or float mask
            modified_mask = np.logical_or(outside_region, land_mask.values > 0)

        ncells_new_land = (modified_mask > 0).sum()
        logger.info(f'  Cells marked as land after modification: {ncells_new_land}')

        # Update dataset with modified mask
        ds_mask[mask_var] = (land_mask.dims, modified_mask)

        # Copy attributes
        ds_mask[mask_var].attrs.update(land_mask.attrs)
        ds_mask[mask_var].attrs['comment'] = (
            'Land mask modified to include regional domain boundaries. '
            'Cells outside the specified domain are marked as land.')

        # Write output
        ds_mask.to_netcdf('land_mask_with_region.nc')

        logger.info('  Modified land mask written to land_mask_with_region.nc')
