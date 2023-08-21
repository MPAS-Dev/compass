from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.translate import center

from compass.model import make_graph_file
from compass.step import Step


class ConvInit(Step):
    """
    A step for creating a mesh for a given resolution in a mesh convergence
    test case.  A child class of this step should then create an appropriate
    initial condition.

    Attributes
    ----------
    resolution : int
        The resolution of the test case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : int
            The resolution of the test case
        """
        super().__init__(test_case=test_case,
                         name='{}km_init'.format(resolution),
                         subdir='{}km/init'.format(resolution))

        for file in ['mesh.nc', 'graph.info']:
            self.add_output_file(file)

        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        resolution = float(self.resolution)

        section = config['mesh_convergence']
        nx_1km = section.getint('nx_1km')
        ny_1km = section.getint('ny_1km')
        nx = int(nx_1km / resolution)
        ny = int(ny_1km / resolution)
        dc = resolution * 1e3
        nonperiodic = section.getboolean('nonperiodic')

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=nonperiodic,
                                       nonperiodic_y=nonperiodic)

        center(ds_mesh)
        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, logger=logger)

        write_netcdf(ds_mesh, 'mesh.nc')
        make_graph_file('mesh.nc', 'graph.info')
