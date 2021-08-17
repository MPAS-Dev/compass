import numpy as np

from mpas_tools.ocean import build_spherical_mesh

from compass.step import Step
from compass.model import make_graph_file


class Mesh(Step):
    """
    A step for creating uniform global meshes

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km
    """

    def __init__(self, test_case, resolution):
        """
        Create a new step

        Parameters
        ----------
        test_case :
            compass.ocean.tests.sphere_transport.correlatedTracers2D.CorrelatedTracers2D

        resolution : int
            The resolution of the (uniform) mesh in km
        """
        super().__init__(test_case=test_case,
                         name='QU{}_mesh'.format(resolution),
                         subdir='QU{}/mesh'.format(resolution))
        for file in ['mesh.nc', 'graph.info']:
            self.add_output_file(filename=file)

        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger

        # only use progress bars if we're not writing to a log file
        use_progress_bar = self.log_filename is None

        # create the base mesh
        cellWidth, lon, lat = self.build_cell_width_lat_lon()
        build_spherical_mesh(cellWidth, lon, lat, out_filename='mesh.nc',
                             logger=logger, use_progress_bar=use_progress_bar)

        make_graph_file(mesh_filename='mesh.nc',
                        graph_filename='graph.info')

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
        constantCellWidth = float(self.resolution)

        nlat = int(180 / dlat) + 1
        nlon = int(360 / dlon) + 1
        lat = np.linspace(-90., 90., nlat)
        lon = np.linspace(-180., 180., nlon)

        cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
        return cellWidth, lon, lat
