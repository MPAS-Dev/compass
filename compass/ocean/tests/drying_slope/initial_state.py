from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.model import run_model
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for drying slope test
    cases
    """
    def __init__(self, test_case, resolution, name='initial_state',
                 coord_type='sigma'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.drying_slope.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name=name, ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.coord_type = coord_type

        self.resolution = resolution

        self.add_namelist_file('compass.ocean.tests.drying_slope',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.drying_slope',
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

        section = config['vertical_grid']
        coord_type = self.coord_type
        thin_film_thickness = section.getfloat('thin_film_thickness') + 1.0e-9
        if coord_type == 'single_layer':
            options = {'config_tidal_boundary_vert_levels': '1',
                       'config_drying_min_cell_height':
                       f'{thin_film_thickness}'}
            self.update_namelist_at_runtime(options)
        else:
            vert_levels = section.getint('vert_levels')
            options = {'config_tidal_boundary_vert_levels': f'{vert_levels}',
                       'config_tidal_boundary_layer_type': f"'{coord_type}'",
                       'config_drying_min_cell_height':
                       f'{thin_film_thickness}'}
            self.update_namelist_at_runtime(options)

        # Determine mesh parameters
        nx = config.getint('drying_slope', 'nx')
        ny = round(28 / self.resolution)
        if self.resolution < 1.:
            ny += 2
        dc = 1e3 * self.resolution

        logger.info(' * Make planar hex mesh')
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
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
