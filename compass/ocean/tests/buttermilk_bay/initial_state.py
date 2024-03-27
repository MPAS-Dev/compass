from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.model import run_model
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for buttermilk bay test
    cases
    """
    def __init__(self, test_case, name, resolution,
                 coord_type='single_layer', wetdry='standard'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.buttermilk_bay.default.Default
            The test case this step belongs to
        """
        self.coord_type = coord_type
        self.resolution = resolution

        super().__init__(test_case=test_case, name=name, ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_namelist_file('compass.ocean.tests.buttermilk_bay',
                               'namelist.init', mode='init')

        if wetdry == 'subgrid':
            self.add_namelist_file('compass.ocean.tests.buttermilk_bay',
                                   'namelist.subgrid.init', mode='init')

        self.add_streams_file('compass.ocean.tests.buttermilk_bay',
                              'streams.init', mode='init')

        self.add_input_file(
            filename='buttermilk_bathy.nc',
            target='buttermilk_bathy.nc',
            database='bathymetry_database')

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'ocean.nc', 'init_mode_forcing_data.nc']:
            self.add_output_file(file)

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        # Set vertical levels
        coord_type = self.coord_type
        if coord_type == 'single_layer':
            options = {'config_buttermilk_bay_vert_levels': '1'}
        else:
            vert_levels = config.getint('vertical_grid', 'vert_levels')
            options = {'config_buttermilk_bay_vert_levels': f'{vert_levels}'}
        self.update_namelist_at_runtime(options)

        # Determine mesh parameters
        Lx = config.getint('buttermilk_bay', 'Lx')
        Ly = config.getint('buttermilk_bay', 'Ly')
        nx = round(Lx / self.resolution)
        ny = round(Ly / self.resolution)
        dc = self.resolution

        logger.info(' * Make planar hex mesh')
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=True)
        logger.info(' * Completed Make planar hex mesh')
        write_netcdf(dsMesh, 'base_mesh.nc')

        logger.info(' * Cull mesh')
        dsMesh = cull(dsMesh, logger=logger)
        logger.info(' * Convert mesh')
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        logger.info(' * Completed Convert mesh')
        write_netcdf(dsMesh, 'culled_mesh.nc')

        run_model(self, namelist='namelist.ocean',
                  streams='streams.ocean')
