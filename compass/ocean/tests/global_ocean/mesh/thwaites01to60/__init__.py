import mpas_tools.mesh.creation.mesh_definition_tools as mdt
import numpy as np
import xarray as xr
from geometric_features import FeatureCollection, GeometricFeatures
from mpas_tools.cime.constants import constants
from mpas_tools.mesh.creation.signed_distance import (
    signed_distance_from_geojson,
)
from scipy import ndimage

from compass.mesh import QuasiUniformSphericalMeshStep


class Thwaites01to60BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating Thwaites 200m-to-60km variable resolution mesh

    Attributes
    ----------
    cell_width : numpy.ndarray
        m x n array of cell width in km

    x, y, z : numpy.ndarray
        m x n arrays defining the sphere
    """

    def setup(self):
        """
        Add geojson files as inputs
        """
        # TODO: Add thwaites_grounding_line.geojson when available
        # self.add_input_file(
        #     filename='thwaites_grounding_line.geojson',
        #     package=self.__module__)

        super().setup()

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            latitude in degrees (length m and between -90 and 90)
        """
        # Get config parameters
        config = self.config
        section = config['thwaites01to60']

        res_gz = section.getfloat('res_gz')
        res_cavity = section.getfloat('res_cavity')
        res_shelf = section.getfloat('res_shelf')
        res_far = section.getfloat('res_far')
        gz_band_halfwidth = section.getfloat('gz_band_halfwidth')

        print('\nCreating Thwaites01to60 mesh with:')
        print(f'  GZ resolution: {res_gz} km')
        print(f'  Cavity resolution: {res_cavity} km')
        print(f'  Shelf resolution: {res_shelf} km')
        print(f'  Far-field resolution: {res_far} km')
        print(f'  GZ band halfwidth: {gz_band_halfwidth} km')

        # Create lat-lon grid for cellWidth
        dlon = 0.1
        dlat = dlon
        earth_radius = constants['SHR_CONST_REARTH']
        nlon = int(360. / dlon) + 1
        nlat = int(180. / dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        # Start with far-field resolution everywhere
        cellWidth = res_far * np.ones((nlat, nlon))

        # Extract and use grounding line from BedMachine
        print('\n  Loading BedMachine to extract grounding line...')
        gz_geojson = self._extract_grounding_line_geojson()

        if gz_geojson is not None:
            print('  Calculating signed distance from grounding line...')
            # Calculate signed distance from grounding line
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            gz_signed_distance = signed_distance_from_geojson(
                gz_geojson, lon_grid, lat_grid, earth_radius,
                max_length=0.25)

            # Convert from meters to km
            gz_signed_distance_km = gz_signed_distance / 1000.0

            # Apply refinement in grounding zone band (±gz_band_halfwidth km)
            # Use smooth tanh transition
            transition_width = 5.0  # km

            # GZ band refinement
            print(f'  Applying GZ refinement (±{gz_band_halfwidth} km band)...')
            gz_mask = 0.5 * (1 + np.tanh(
                (np.abs(gz_signed_distance_km) - gz_band_halfwidth) / transition_width))
            cellWidth = res_gz * (1 - gz_mask) + cellWidth * gz_mask

            # Cavity refinement (ocean side, signed distance < 0)
            # Refine 100km into cavity from grounding line
            cavity_extent = 100.0  # km
            print(f'  Applying cavity refinement ({cavity_extent} km from GL)...')
            cavity_mask = 0.5 * (1 + np.tanh(
                (-gz_signed_distance_km - cavity_extent) / transition_width))
            # Only apply on ocean side
            cavity_mask = np.where(gz_signed_distance_km < 0, cavity_mask, 1.0)
            cellWidth = np.minimum(cellWidth,
                                  res_cavity * (1 - cavity_mask) + cellWidth * cavity_mask)

            # Ice shelf refinement (farther into cavity)
            shelf_extent = 200.0  # km
            print(f'  Applying shelf refinement ({shelf_extent} km from GL)...')
            shelf_mask = 0.5 * (1 + np.tanh(
                (-gz_signed_distance_km - shelf_extent) / transition_width))
            shelf_mask = np.where(gz_signed_distance_km < 0, shelf_mask, 1.0)
            cellWidth = np.minimum(cellWidth,
                                  res_shelf * (1 - shelf_mask) + cellWidth * shelf_mask)

        else:
            print('  WARNING: Could not extract grounding line from BedMachine')
            print('  Falling back to Gaussian approximation around Thwaites')
            # Fall back to simple Gaussian refinement
            cellWidth = self._apply_gaussian_refinement(
                cellWidth, lon, lat, res_gz, res_cavity, res_shelf)

        print(f'  CellWidth range: {cellWidth.min():.2f} - {cellWidth.max():.2f} km')

        return cellWidth, lon, lat

    def _extract_grounding_line_geojson(self):
        """
        Extract grounding line from BedMachine and create a geojson

        Returns
        -------
        fc : geometric_features.FeatureCollection or None
            Feature collection containing grounding line polygon(s)
        """
        try:
            # Try to find BedMachine in the database
            # The file should be specified in the config
            config = self.config
            bedmachine_file = None

            # Check if we can find BedMachine file from compass database
            # For now, try a few standard locations
            import os
            possible_paths = [
                '/global/cfs/cdirs/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/BedMachineAntarctica-v3.nc',
                '/usr/projects/climate/SHARED_CLIMATE/mpas_standalonedata/mpas-ocean/bathymetry_database/BedMachineAntarctica-v3.nc',
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    bedmachine_file = path
                    break

            if bedmachine_file is None:
                print('    Could not find BedMachine file in standard locations')
                return None

            print(f'    Loading BedMachine from: {bedmachine_file}')
            ds_bed = xr.open_dataset(bedmachine_file)

            # Extract ice thickness and bed elevation
            # BedMachine convention: bed < 0 is below sea level
            thickness = ds_bed.thickness.values
            bed = ds_bed.bed.values

            # Get coordinates
            x_bed = ds_bed.x.values
            y_bed = ds_bed.y.values

            # Convert polar stereographic to lat-lon
            # BedMachine uses EPSG:3031 (Antarctic Polar Stereographic)
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:3031", "EPSG:4326")

            # Create coordinate grids (subsample for efficiency)
            skip = 5  # Use every 5th point
            x_grid, y_grid = np.meshgrid(x_bed[::skip], y_bed[::skip])
            lat_bed, lon_bed = transformer.transform(x_grid.flatten(), y_grid.flatten())
            lat_bed = lat_bed.reshape(x_grid.shape)
            lon_bed = lon_bed.reshape(x_grid.shape)

            # Subsample thickness and bed
            thickness_sub = thickness[::skip, ::skip]
            bed_sub = bed[::skip, ::skip]

            # Calculate flotation criterion
            # Ice is grounded when: thickness > -bed * (ρ_ocean / ρ_ice)
            # (only where bed < 0)
            rho_ocean = 1028.0
            rho_ice = 918.0

            print(f'    Calculating flotation (ρ_ocean={rho_ocean}, ρ_ice={rho_ice})...')
            thickness_flotation = np.where(bed_sub < 0,
                                          -bed_sub * (rho_ocean / rho_ice),
                                          np.inf)  # Always grounded if bed > 0

            # Mask for grounded ice
            grounded = thickness_sub > thickness_flotation

            # Find grounding line as edge of grounded region
            # Use morphological gradient to find boundaries
            print('    Identifying grounding line cells...')
            from scipy.ndimage import binary_dilation, binary_erosion

            # Grounding line = grounded cells adjacent to floating cells
            grounded_dilated = binary_dilation(grounded)
            grounded_eroded = binary_erosion(grounded)
            gl_mask = grounded_dilated & ~grounded_eroded

            # Focus on Thwaites/Amundsen region
            # Thwaites: roughly 75°S, 106°W to 74°S, 104°W
            amundsen_mask = ((lat_bed >= -77) & (lat_bed <= -73) &
                            (lon_bed >= -120) & (lon_bed <= -95))
            gl_mask = gl_mask & amundsen_mask

            if np.sum(gl_mask) == 0:
                print('    No grounding line cells found in Amundsen region')
                return None

            print(f'    Found {np.sum(gl_mask)} grounding line cells')

            # Extract grounding line coordinates
            gl_lons = lon_bed[gl_mask]
            gl_lats = lat_bed[gl_mask]

            # Create a simple polygon/multipoint feature
            # For simplicity, create a buffered multipoint
            from shapely.geometry import MultiPoint, Point
            import geojson

            points = [Point(lon, lat) for lon, lat in zip(gl_lons, gl_lats)]
            multipoint = MultiPoint(points)

            # Buffer by ~2km to create polygon
            # At ~75°S, 1 degree ≈ 30 km, so 2km ≈ 0.067 degrees
            buffer_deg = 0.1
            gl_polygon = multipoint.buffer(buffer_deg)

            # Convert to geojson
            feature = geojson.Feature(geometry=gl_polygon, properties={'name': 'Thwaites_GL'})
            feature_collection = geojson.FeatureCollection([feature])

            # Write to temp file and read back with geometric_features
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
                geojson.dump(feature_collection, f)
                temp_geojson = f.name

            from geometric_features import read_feature_collection
            fc = read_feature_collection(temp_geojson)

            # Clean up temp file
            os.remove(temp_geojson)

            print('    Successfully created grounding line feature collection')
            return fc

        except Exception as e:
            print(f'    Error extracting grounding line: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _apply_gaussian_refinement(self, cellWidth, lon, lat,
                                   res_gz, res_cavity, res_shelf):
        """
        Apply simple Gaussian refinement around Thwaites location (fallback)
        """
        thwaites_lat = -75.0
        thwaites_lon = -106.0

        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Simple distance-based refinement
        dist_lat = (lat_grid - thwaites_lat) ** 2
        dist_lon = ((lon_grid - thwaites_lon) * np.cos(np.radians(lat_grid))) ** 2
        approx_dist_deg = np.sqrt(dist_lat + dist_lon)

        gz_dist = 0.2
        cavity_dist = 1.0
        shelf_dist = 3.0
        transition_width = 0.1

        # Apply refinements
        mask_gz = 0.5 * (1 + np.tanh((approx_dist_deg - gz_dist) / transition_width))
        cellWidth = res_gz * (1 - mask_gz) + cellWidth * mask_gz

        mask_cavity = 0.5 * (1 + np.tanh((approx_dist_deg - cavity_dist) / transition_width))
        cellWidth = np.minimum(cellWidth,
                              res_cavity * (1 - mask_cavity) + cellWidth * mask_cavity)

        mask_shelf = 0.5 * (1 + np.tanh((approx_dist_deg - shelf_dist) / transition_width))
        cellWidth = np.minimum(cellWidth,
                              res_shelf * (1 - mask_shelf) + cellWidth * mask_shelf)

        return cellWidth
