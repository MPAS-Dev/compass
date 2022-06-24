from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh for drying slope test cases

    Attributes
    ----------
    coord_type : {'sigma', 'single_layer'}
        The type of vertical coordinate
    """
    def __init__(self, test_case, coord_type):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.drying_slope.default.Default
            The test case this step belongs to

        coord_type : {'sigma', 'single_layer'}
            The type of vertical coordinate
        """
        super().__init__(test_case=test_case, name='mesh', ntasks=1,
                         min_tasks=1, openmp_threads=1)
        self.coord_type = coord_type

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger

        config = self.config

        section = config['drying_slope']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        logger.info(' * Make planar hex mesh')
        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=False,
                                       nonperiodic_y=True)
        logger.info(' * Completed Make planar hex mesh')
        write_netcdf(ds_mesh, 'base_mesh.nc')

        logger.info(' * Cull mesh')
        ds_mesh = cull(ds_mesh, logger=logger)
        logger.info(' * Convert mesh')
        ds_mesh = convert(ds_mesh, graphInfoFileName='culled_graph.info',
                          logger=logger)
        logger.info(' * Completed Convert mesh')
        write_netcdf(ds_mesh, 'culled_mesh.nc')
