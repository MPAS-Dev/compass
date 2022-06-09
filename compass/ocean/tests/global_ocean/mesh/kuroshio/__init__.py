import numpy as np

import mpas_tools.mesh.creation.mesh_definition_tools as mdt
from mpas_tools.mesh.creation.signed_distance import \
    signed_distance_from_geojson
from geometric_features import read_feature_collection
from mpas_tools.cime.constants import constants

from compass.mesh import QuasiUniformSphericalMeshStep


class KuroshioBaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating Kuroshio meshes
    """
    def setup(self):
        """
        Add some input files
        """
        self.add_input_file(filename='wbc_rectangle3-1.geojson',
                            package=self.__module__)
        self.add_input_file(filename='wbc_rectangle3-2.geojson',
                            package=self.__module__)
        self.add_input_file(filename='wbc_rectangle3-3.geojson',
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
        config = self.config
        dlon = 0.1
        dlat = dlon
        earth_radius = constants['SHR_CONST_REARTH']
        nlon = int(360./dlon) + 1
        nlat = int(180./dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        cellWidth = mdt.EC_CellWidthVsLat(lat, cellWidthEq=30.,
                                          cellWidthMidLat=60.,
                                          cellWidthPole=35.,
                                          latPosEq=7.5, latWidthEq=3.0)

        _, cellWidth = np.meshgrid(lon, cellWidth)

        fc1 = read_feature_collection('wbc_rectangle3-1.geojson')

        ks_signed_distance1 = signed_distance_from_geojson(fc1, lon, lat,
                                                          earth_radius,
                                                          max_length=0.25)

        trans_width = 400e3
        trans_start = 0
        dx_min = config.getfloat('global_ocean', 'min_res')

        weights = 0.5 * (1 + np.tanh((ks_signed_distance1 - trans_start) /
                                     trans_width))

        cellWidth = dx_min * (1 - weights) + cellWidth * weights


        fc2 = read_feature_collection('wbc_rectangle3-2.geojson')

        ks_signed_distance2 = signed_distance_from_geojson(fc2, lon, lat,
                                                          earth_radius,
                                                          max_length=0.25)

        trans_width = 400e3
        trans_start = 0
        dx_min = 40.

        weights = 0.5 * (1 + np.tanh((ks_signed_distance2 - trans_start) /
                                     trans_width))

        cellWidth = dx_min * (1 - weights) + cellWidth * weights


        fc3 = read_feature_collection('wbc_rectangle3-3.geojson')

        ks_signed_distance3 = signed_distance_from_geojson(fc3, lon, lat,
                                                          earth_radius,
                                                          max_length=0.25)

        trans_width = 400e3
        trans_start = 0
        dx_min = 30.

        weights = 0.5 * (1 + np.tanh((ks_signed_distance3 - trans_start) /
                                     trans_width))

        cellWidth = dx_min * (1 - weights) + cellWidth * weights

        return cellWidth, lon, lat
