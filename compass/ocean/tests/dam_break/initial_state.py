import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.model import run_model
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for dam break test
    cases
    """
    def __init__(self, test_case, use_lts):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.dam_break.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='initial_state', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.use_lts = use_lts

        if use_lts:
            self.add_namelist_file('compass.ocean.tests.dam_break.lts',
                                   'namelist.init', mode='init')
        else:
            self.add_namelist_file('compass.ocean.tests.dam_break',
                                   'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.dam_break',
                              'streams.init', mode='init')

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'ocean.nc']:
            self.add_output_file(file)

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['dam_break']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        self.update_namelist_at_runtime(
            {'config_dam_break_dc': f'{dc}'})

        logger.info(' * Make planar hex mesh')
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=True)
        logger.info(' * Completed Make planar hex mesh')
        write_netcdf(dsMesh, 'base_mesh.nc')

        logger.info(' * Cull mesh boundaries')
        dsMesh = cull(dsMesh, logger=logger)
        logger.info(' * Convert mesh')
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        logger.info(' * Completed Convert mesh')
        write_netcdf(dsMesh, 'culled_mesh.nc')

        run_model(self, namelist='namelist.ocean',
                  streams='streams.ocean')

        postrun_data = xarray.open_dataset('ocean.nc')
        logger.info(' * Cull mesh dam boundaries')
        postrun_data_cull = cull(postrun_data, logger=logger)
        logger.info(' * Convert mesh')
        postrun_data_cull = convert(postrun_data_cull,
                                    graphInfoFileName='culled_graph.info',
                                    logger=logger)
        write_netcdf(postrun_data_cull, 'culled_mesh.nc')

        run_model(self, namelist='namelist.ocean',
                  streams='streams.ocean')
