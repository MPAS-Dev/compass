import mpas_tools.mesh.creation.mesh_definition_tools as mdt
import numpy as np

from compass.mesh import QuasiUniformSphericalMeshStep


class RRS6to18BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating RRS6to18 and RRSwISC6to18 meshes
    """

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

        dlon = 10.
        dlat = 0.1
        nlon = int(360. / dlon) + 1
        nlat = int(180. / dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        cellWidthVsLat = mdt.RRS_CellWidthVsLat(lat, cellWidthEq=18.,
                                                cellWidthPole=6.)
        cellWidth = np.outer(cellWidthVsLat, np.ones([1, lon.size]))

        return cellWidth, lon, lat
