import numpy as np

import mpas_tools.mesh.creation.mesh_definition_tools as mdt
from mpas_tools.mesh.creation.signed_distance import \
    signed_distance_from_geojson
from geometric_features import read_feature_collection
from mpas_tools.cime.constants import constants

from compass.mesh import QuasiUniformSphericalMeshStep


class SO12to60BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating SO12to60 meshes
    """
    def setup(self):
        """
        Add some input files
        """

        self.add_input_file(filename='atlantic.geojson',
                            package=self.__module__)

        self.add_input_file(filename='high_res_region.geojson',
                            package=self.__module__)

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

        cellWidthSouth = mdt.EC_CellWidthVsLat(lat, cellWidthEq=30.,
                                               cellWidthMidLat=45.,
                                               cellWidthPole=45.,
                                               latPosEq=7.5, latWidthEq=3.0)

        cellWidthNorth = mdt.EC_CellWidthVsLat(lat, cellWidthEq=30.,
                                               cellWidthMidLat=60.,
                                               cellWidthPole=35.,
                                               latPosEq=7.5, latWidthEq=3.0)

        # Transition at Equator
        latTransition = 0.0
        latWidthTransition = 2.5
        cellWidthVsLat = mdt.mergeCellWidthVsLat(
            lat,
            cellWidthSouth,
            cellWidthNorth,
            latTransition,
            latWidthTransition)

        _, cellWidth = np.meshgrid(lon, cellWidthVsLat)

        cellWidthAtlantic = mdt.EC_CellWidthVsLat(lat, cellWidthEq=30.,
                                                  cellWidthMidLat=30.,
                                                  cellWidthPole=35.,
                                                  latPosEq=7.5, latWidthEq=3.0)

        cellWidthAtlantic = mdt.mergeCellWidthVsLat(
            lat,
            cellWidthSouth,
            cellWidthAtlantic,
            latTransition,
            latWidthTransition)

        _, cellWidthAtlantic = np.meshgrid(lon, cellWidthAtlantic)

        fc = read_feature_collection('atlantic.geojson')

        atlantic_signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25)

        trans_width = 400e3
        trans_start = 0.
        weights = 0.5 * (1 + np.tanh((atlantic_signed_distance - trans_start) /
                                     trans_width))

        cellWidth = cellWidthAtlantic * (1 - weights) + cellWidth * weights

        fc = read_feature_collection('high_res_region.geojson')

        so_signed_distance = signed_distance_from_geojson(fc, lon, lat,
                                                          earth_radius,
                                                          max_length=0.25)

        # Equivalent to 20 degrees latitude
        trans_width = 1600e3
        trans_start = 500e3
        dx_min = 12.

        weights = 0.5 * (1 + np.tanh((so_signed_distance - trans_start) /
                                     trans_width))

        cellWidth = dx_min * (1 - weights) + cellWidth * weights

        return cellWidth, lon, lat
