import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for drying slope test
    cases
    """
    def __init__(self, test_case, resolution, name='initial_state',
                 baroclinic=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.drying_slope.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name=name, ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.resolution = resolution

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc', 'forcing.nc']:
            self.add_output_file(file)

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        resolution = self.resolution

        # Fetch config options
        section = config['vertical_grid']
        thin_film_thickness = section.getfloat('thin_film_thickness') + 1.0e-9
        vert_levels = section.getint('vert_levels')

        section = config['drying_slope']
        nx = section.getint('nx')
        domain_length = section.getfloat('ly') * 1e3
        drying_length = section.getfloat('ly_analysis') * 1e3
        plug_width_frac = section.getfloat('plug_width_frac')
        right_bottom_depth = section.getfloat('right_bottom_depth')
        left_bottom_depth = section.getfloat('left_bottom_depth')
        plug_temperature = section.getfloat('plug_temperature')
        background_temperature = section.getfloat('background_temperature')
        background_salinity = section.getfloat('background_salinity')
        coriolis_parameter = section.getfloat('coriolis_parameter')

        # Check config options
        if domain_length < drying_length:
            raise ValueError('Domain is not long enough to capture wetting '
                             'front')
        if right_bottom_depth < left_bottom_depth:
            raise ValueError('Right boundary must be deeper than left '
                             'boundary')

        # Determine mesh parameters
        dc = 1e3 * resolution
        ny = round(domain_length / dc)
        # This is just for consistency with previous implementations and could
        # be removed
        if resolution < 1.:
            ny += 2
        ny = 2 * round(ny / 2)

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

        ds = ds_mesh.copy()
        ds_forcing = ds_mesh.copy()

        y_min = ds_mesh.yCell.min()
        y_max = ds_mesh.yCell.max()
        dc_edge_min = ds_mesh.dcEdge.min()

        y_cell = ds.yCell
        max_level_cell = vert_levels
        bottom_depth = (right_bottom_depth - (y_max - y_cell) / drying_length *
                        (right_bottom_depth - left_bottom_depth))
        ds['bottomDepth'] = bottom_depth
        # Set the water column to dry everywhere
        ds['ssh'] = -bottom_depth + thin_film_thickness * max_level_cell
        # We don't use config_tidal_forcing_monochromatic_baseline because the
        # default value doesn't alter the initial state
        init_vertical_coord(config, ds)

        plug_width = domain_length * plug_width_frac
        y_plug_boundary = y_min + plug_width
        ds['temperature'] = xr.where(y_cell < y_plug_boundary,
                                     plug_temperature, background_temperature)
        ds['tracer1'] = xr.where(y_cell < y_plug_boundary, 1.0, 0.0)
        ds['salinity'] = background_salinity * xr.ones_like(y_cell)
        normalVelocity = xr.zeros_like(ds_mesh.xEdge)
        normalVelocity, _ = xr.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)
        ds['normalVelocity'] = normalVelocity
        ds['fCell'] = coriolis_parameter * xr.ones_like(ds.xCell)
        ds['fEdge'] = coriolis_parameter * xr.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xr.ones_like(ds.xVertex)

        write_netcdf(ds, 'initial_state.nc')

        # Define the tidal boundary condition over 1-cell width
        y_tidal_boundary = y_max - dc_edge_min / 2.
        tidal_forcing_mask = xr.where(y_cell > y_tidal_boundary, 1.0, 0.0)
        if tidal_forcing_mask.sum() <= 0:
            raise ValueError('Input mask for tidal case is not set!')
        ds_forcing['tidalInputMask'] = tidal_forcing_mask
        write_netcdf(ds_forcing, 'forcing.nc')
