import numpy as np

from compass.ocean.tests.global_ocean.mesh.mesh import MeshStep


class QU240Mesh(MeshStep):
    """
    A step for creating QU240 and QUwISC240 meshes
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
                         mesh_config_filename='qu240.cfg')

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
        dlat = dlon
        constantCellWidth = 240.

        nlat = int(180/dlat) + 1
        nlon = int(360/dlon) + 1
        lat = np.linspace(-90., 90., nlat)
        lon = np.linspace(-180., 180., nlon)

        cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
        return cellWidth, lon, lat
