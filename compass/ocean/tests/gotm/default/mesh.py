from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh for General Ocean Turbulence Model (GOTM) test
    cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.gotm.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='mesh', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_namelist_file('compass.ocean.tests.gotm.default',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.gotm.default',
                              'streams.init', mode='init')

        for file in ['mesh.nc', 'graph.info']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['gotm']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        ds_mesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                       nonperiodic_x=False,
                                       nonperiodic_y=False)
        write_netcdf(ds_mesh, 'grid.nc')

        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, graphInfoFileName='graph.info',
                          logger=logger)
        write_netcdf(ds_mesh, 'mesh.nc')
