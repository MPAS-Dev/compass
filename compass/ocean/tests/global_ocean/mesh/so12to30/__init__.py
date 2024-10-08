import numpy as np
from geometric_features import read_feature_collection
from mpas_tools.cime.constants import constants
from mpas_tools.mesh.creation.signed_distance import (
    signed_distance_from_geojson,
)

from compass.mesh import QuasiUniformSphericalMeshStep


class SO12to30BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating SO12to30 meshes
    """
    def setup(self):
        """
        Add some input files
        """

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
        nlon = int(360. / dlon) + 1
        nlat = int(180. / dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        # start with a uniform 30 km background resolution
        dx_max = 30.
        cell_width = dx_max * np.ones((nlat, nlon))

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

        cell_width = dx_min * (1 - weights) + cell_width * weights

        return cell_width, lon, lat
