import xarray
import numpy

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.step import Step
from compass.model import run_model


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for drying slope test
    cases
    """
    def __init__(self, test_case, coord_type='sigma'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.drying_slope.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='initial_state', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.coord_type = coord_type

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

        config = self.config
        section = config['vertical_grid']
        coord_type = self.coord_type
        if coord_type == 'single_layer':
            options = {'config_tidal_boundary_vert_levels': '1'}
            self.update_namelist_at_runtime(options)
        else:
            vert_levels = section.get('vert_levels')
            options = {'config_tidal_boundary_vert_levels': f'{vert_levels}',
                       'config_tidal_boundary_layer_type': f"'{coord_type}'"}
            self.update_namelist_at_runtime(options)

        section = config['drying_slope']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

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
