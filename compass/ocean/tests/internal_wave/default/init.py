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
        super().__init__(test_case=test_case, name='init', cores=1,
                         min_cores=1, threads=1)

        self.add_namelist_file('compass.ocean.tests.internal_wave.default',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.internal_wave.default',
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

        section = config['internal_wave']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        logger.info(' * Make planar hex mesh')
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=False)
        logger.info(' * Completed Make planar hex mesh')
        write_netcdf(dsMesh, 'grid.nc')

        logger.info(' * Cull mesh')
        dsMesh = cull(dsMesh, logger=logger)
        logger.info(' * Convert mesh')
        dsMesh = convert(dsMesh, graphInfoFileName='graph.info',
                         logger=logger)
        logger.info(' * Completed Convert mesh')
        write_netcdf(dsMesh, 'mesh.nc')

        replacements = dict()
        replacements['config_periodic_planar_vert_levels'] = \
            config.get('internal_wave', 'vert_levels')
        replacements['config_periodic_planar_bottom_depth'] = \
            config.get('internal_wave', 'bottom_depth')
        self.update_namelist_at_runtime(options=replacements)

        logger.info(' * Run model')
        run_model(self)
        logger.info(' * Completed Run model')
