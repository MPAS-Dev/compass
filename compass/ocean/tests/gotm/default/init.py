from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step
from compass.model import run_model


class Init(Step):
    """
    A step for creating a mesh and initial condition for General Ocean
    Turbulence Model (GOTM) test cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.gotm.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='init', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_namelist_file('compass.ocean.tests.gotm.default',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.gotm.default',
                              'streams.init', mode='init')

        self.add_model_as_input()

        for file in ['mesh.nc', 'graph.info', 'ocean.nc']:
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

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=False)
        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'mesh.nc')

        replacements = dict()
        replacements['config_periodic_planar_vert_levels'] = \
            config.get('gotm', 'vert_levels')
        replacements['config_periodic_planar_bottom_depth'] = \
            config.get('gotm', 'bottom_depth')
        self.update_namelist_at_runtime(options=replacements)

        run_model(self)
