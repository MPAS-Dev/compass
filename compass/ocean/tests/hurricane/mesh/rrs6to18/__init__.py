import mpas_tools.mesh.creation.mesh_definition_tools as mdt
import numpy as np

from compass.ocean.mesh.floodplain import FloodplainMeshStep


class RRS6to18BaseMesh(FloodplainMeshStep):
    """
    A step for creating RRS6to18 and RRSwISC6to18 meshes
    """

    def __init__(self, test_case, pixel, name='base_mesh', subdir=None,
                 elev_file='RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc'):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        self.elev_file = elev_file
        pixel_path = pixel.path

        self.add_input_file(
            filename='bathy.nc',
            work_dir_target=f'{pixel_path}/{elev_file}')

        self.add_input_file(filename='mab_floodplain.geojson',
                            package=self.__module__)

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
