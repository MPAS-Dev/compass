import netCDF4 as nc4
import numpy as np
from geometric_features import read_feature_collection
from inpoly import inpoly2
from mpas_tools.mesh.creation.signed_distance import _add_poly

import compass.ocean.tests.tides.dem.dem_remap as dem_remap
from compass.mesh.spherical import QuasiUniformSphericalMeshStep


class FloodplainMeshStep(QuasiUniformSphericalMeshStep):
    """
    A step for creating a global MPAS-Ocean mesh that includes variables
    needed for preserving a floodplain

    preserve_floodplain : bool
        Whether the mesh includes land cells
    """

    def __init__(self, test_case, name='base_mesh', subdir=None,
                 cell_width=None, preserve_floodplain=True):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str
            the name of the step

        subdir : {str, None}
            the subdirectory for the step.  The default is ``name``

        cell_width : float, optional
            The approximate cell width in km of the mesh if constant resolution

        preserve_floodplain : bool, optional
            Whether the mesh includes land cells
        """

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cell_width=cell_width)

        self.preserve_floodplain = preserve_floodplain

        self.add_input_file(filename='earth_relief_15s.nc',
                            target='SRTM15_plus_earth_relief_15s.nc',
                            database='bathymetry_database')

        pixel_path = test_case.steps['pixel'].path
        pixel_file = f'{pixel_path}/RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc'
        self.add_input_file(
            filename='bathy.nc',
            work_dir_target=pixel_file)

        self.bottomDepth_varname = 'bottomDepthObserved'

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        config = self.config

        mesh_filename = config.get('spherical_mesh', 'mpas_mesh_filename')

        dem_remap.dem_remap('bathy.nc', mesh_filename)

        # Create new NetCDF variables in mesh file, if necessary
        nc_mesh = nc4.Dataset(mesh_filename, 'r+')
        nc_vars = nc_mesh.variables.keys()
        if 'bottomDepthObserved' not in nc_vars:
            nc_mesh.createVariable('bottomDepthObserved', 'f8', ('nCells'))

        # Write to mesh file
        nc_mesh.variables['bottomDepthObserved'][:] = \
            nc_mesh.variables['bed_elevation'][:]
        nc_mesh.close()

        if self.preserve_floodplain:
            floodplain_elevation = config.getfloat('spherical_mesh',
                                                   'floodplain_elevation')
            min_depth_outside_floodplain = config.getfloat(
                'spherical_mesh',
                'min_depth_outside_floodplain')
            if config.has_option('spherical_mesh', 'floodplain_resolution'):
                floodplain_resolution = config.getfloat(
                    'spherical_mesh',
                    'floodplain_resolution')
            else:
                floodplain_resolution = 1e10
            if config.has_option('spherical_mesh', 'floodplain_geojson'):
                floodplain_geojson = config.get(
                    'spherical_mesh',
                    'floodplain_geojson')
                floodplain_region = read_feature_collection(floodplain_geojson)
            else:
                floodplain_region = None

            self.inject_preserve_floodplain(
                mesh_file=mesh_filename,
                floodplain_elevation=floodplain_elevation,
                floodplain_resolution=floodplain_resolution,
                floodplain_region=floodplain_region,
                min_depth_outside_floodplain=min_depth_outside_floodplain)

    def inject_preserve_floodplain(self, mesh_file, floodplain_elevation,
                                   floodplain_resolution=1e10,
                                   floodplain_region=None,
                                   min_depth_outside_floodplain=5.0):

        nc_mesh = nc4.Dataset(mesh_file, 'r+')
        nc_vars = nc_mesh.variables.keys()

        if 'regionCellMasks' not in nc_vars:
            nc_mesh.createDimension('nRegions', 1)
            nc_mesh.createVariable('regionCellMasks', 'i',
                                   ('nCells', 'nRegions'))
        nc_mesh.variables['regionCellMasks'][:] = 0.0

        if 'transectCellMasks' not in nc_vars:
            nc_mesh.createDimension('nTransects', 1)
            nc_mesh.createVariable('transectCellMasks', 'i',
                                   ('nCells', 'nTransects'))
        nc_mesh.variables['transectCellMasks'][:] = 0.0

        floodplain = self.find_floodplain(mesh_file, floodplain_elevation,
                                          floodplain_resolution,
                                          floodplain_region)

        nc_mesh.variables['regionCellMasks'][:] = floodplain

        nc_mesh.close()

    def find_floodplain(self, mesh_file, floodplain_elevation,
                        floodplain_resolution,
                        floodplain_region):

        nc_mesh = nc4.Dataset(mesh_file, 'r+')

        bottomDepth = self.bottomDepth_varname
        h = 2.0 * np.sqrt(nc_mesh.variables['areaCell'][:] / np.pi) / 1000.0
        floodplain = np.logical_and(
            h < floodplain_resolution,
            nc_mesh.variables[bottomDepth][:] < floodplain_elevation)
        if floodplain_region:
            print("adding floodplain region")
            nodes = list()
            edges = list()
            for feature in floodplain_region.features:
                if feature['geometry']['type'] == 'Polygon':
                    for poly in feature['geometry']['coordinates']:
                        _add_poly(poly, edges, nodes)

                elif feature['geometry']['type'] == 'MultiPolygon':
                    for mpoly in feature['geometry']['coordinates']:
                        for poly in mpoly:
                            _add_poly(poly, edges, nodes)

            nodes = np.array(nodes)
            edges = np.array(edges)
            print(nodes)
            print(edges)

            lon = np.degrees(nc_mesh.variables['lonCell'][:])
            lon = np.mod(lon + 180.0, 360.0) - 180.0
            lat = np.degrees(nc_mesh.variables['latCell'][:])

            points = np.vstack([lon, lat]).T

            floodplain_mask, _ = inpoly2(points, nodes, edges)
            floodplain_mask = 1 * floodplain_mask
            print(np.max(floodplain_mask))
            print(np.min(floodplain_mask))

        else:
            floodplain_mask = 1
        floodplain = floodplain_mask * floodplain

        nc_mesh.close()

        return floodplain
