import numpy as np
import mpas_tools.mesh.creation.mesh_definition_tools as mdt

from compass.ocean.tests.global_ocean.mesh.mesh import MeshStep


class EC30to60Mesh(MeshStep):
    """
    A step for creating EC30to60 and ECwISC30to60 meshes
    """
    def __init__(self, test_case, mesh_name, with_ice_shelf_cavities):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh_name : str
            The name of the mesh

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, mesh_name, with_ice_shelf_cavities,
                         package=self.__module__,
                         mesh_config_filename='ec30to60.cfg')

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
        nlon = int(360./dlon) + 1
        nlat = int(180./dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        cellWidthVsLat = mdt.EC_CellWidthVsLat(lat)
        cellWidth = np.outer(cellWidthVsLat, np.ones([1, lon.size]))

        return cellWidth, lon, lat
