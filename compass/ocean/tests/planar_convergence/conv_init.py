from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.translate import center
from mpas_tools.io import write_netcdf

from compass.step import Step
from compass.model import make_graph_file


class ConvInit(Step):
    """
    A step for creating a mesh for a given resolution in a planar convergence
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
        config = self.config
        resolution = float(self.resolution)

        section = config['planar_convergence']
        nx_1km = section.getint('nx_1km')
        ny_1km = section.getint('ny_1km')
        nx = int(nx_1km/resolution)
        ny = int(ny_1km/resolution)
        dc = resolution*1e3

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=False,
                                       nonperiodic_y=False)

        center(ds_mesh)

        write_netcdf(ds_mesh, 'mesh.nc')
        make_graph_file('mesh.nc', 'graph.info')
