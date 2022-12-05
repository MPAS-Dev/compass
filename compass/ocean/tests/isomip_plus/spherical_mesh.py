import numpy as np
import xarray as xr
import pyproj

from mpas_tools.mesh.creation.signed_distance import \
    signed_distance_from_geojson
from geometric_features import FeatureCollection
from mpas_tools.cime.constants import constants

from compass.mesh import QuasiUniformSphericalMeshStep


class SphericalMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating an ISOMIP+ mesh that is a small region on a sphere
    """
    def setup(self):
        """
        Add some input files
        """
        self.add_input_file(
            filename='input_geometry_processed.nc',
            target='../process_geom/input_geometry_processed.nc')

        self.add_output_file('base_mesh_with_xy.nc')

        super().setup()

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """

        dlon = 0.1
        dlat = dlon
        earth_radius = constants['SHR_CONST_REARTH']
        nlon = int(360./dlon) + 1
        nlat = int(180./dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        # this is the width of cells (in km) on the globe outside the domain of
        # interest, set to a coarse value to speed things up
        background_width = 100.

        lat0 = self.config.getfloat('spherical_mesh', 'lat0')
        fc = _make_feature(lat0)
        fc.to_geojson('isomip_plus_high_res.geojson')

        signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25)

        # this is a distance (in m) over which the resolution coarsens outside
        # the domain of interest plus buffer
        trans_width = 1000e3
        weights = np.maximum(0., np.minimum(1., signed_distance/trans_width))

        cell_width = (self.cell_width * (1 - weights)
                      + background_width * weights)

        return cell_width, lon, lat

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        ds = xr.open_dataset('base_mesh.nc')
        lat0 = self.config.getfloat('spherical_mesh', 'lat0')
        add_isomip_plus_xy(ds, lat0)
        ds.to_netcdf('base_mesh_with_xy.nc')


def add_isomip_plus_xy(ds, lat0):
    """
    Add x and y coordinates from a stereographic projection to a mesh on a
    sphere

    Parameters
    ----------
    ds : xarray.Dataset
        The MPAS mesh on a sphere

    lat0 : float
        The latitude of the origin of the local stereographic project
    """
    projection, lat_lon_projection = _get_projections(lat0)
    transformer = pyproj.Transformer.from_proj(lat_lon_projection,
                                               projection)
    lon = np.rad2deg(ds.lonCell.values)
    lat = np.rad2deg(ds.latCell.values)

    x, y = transformer.transform(lon, lat)

    ds['xIsomipCell'] = ('nCells', x)
    ds['yIsomipCell'] = ('nCells', y)

    lon = np.rad2deg(ds.lonVertex.values)
    lat = np.rad2deg(ds.latVertex.values)

    x, y = transformer.transform(lon, lat)

    ds['xIsomipVertex'] = ('nVertices', x)
    ds['yIsomipVertex'] = ('nVertices', y)


def _get_projections(lat0):
    projection = pyproj.Proj(f'+proj=stere +lon_0=0 +lat_0={lat0} '
                             f'+lat_ts={lat0} +x_0=0.0 +y_0=0.0 +ellps=WGS84')
    lat_lon_projection = pyproj.Proj(proj='latlong', datum='WGS84')

    return projection, lat_lon_projection


def _make_feature(lat0):
    # a box with a buffer of 80 km surrounding the are of interest
    # (0 <= x <= 800) and (0 <= y <= 80)
    bounds = 1e3*np.array((-80., 880., -80., 160.))
    projection, lat_lon_projection = _get_projections(lat0)
    transformer = pyproj.Transformer.from_proj(projection,
                                               lat_lon_projection)

    x = [bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]]
    y = [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]]
    lon, lat = transformer.transform(x, y)

    coordinates = [[[lon[index], lat[index]] for index in range(len(lon))]]

    features = [
        {
            "type": "Feature",
            "properties": {
                "name": "ISOMIP+ high res region",
                "component": "ocean",
                "object": "region",
                "author": "Xylar Asay-Davis"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }
    ]

    fc = FeatureCollection(features=features)

    return fc
