import xarray
import numpy

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for internal wave test
    cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.internal_wave.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='initial_state', cores=1,
                         min_cores=1, threads=1)

        self.add_namelist_file('compass.ocean.tests.internal_wave',
                               'namelist.init')

        self.add_streams_file('compass.ocean.tests.internal_wave',
                              'streams.init')

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'ocean.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        replacements = dict()
        replacements['config_periodic_planar_vert_levels'] = \
            config.getfloat('vertical_grid', 'vert_levels')
        replacements['config_periodic_planar_bottom_depth'] = \
            config.getfloat('vertical_grid', 'bottom_depth')
        self.update_namelist_at_runtime(options=replacements)

        section = config['vertical_grid']
        vert_levels = section.getint('vert_levels')
        bottom_depth = section.getfloat('bottom_depth')

        section = config['internal_wave']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')
        use_distances = section.getboolean('use_distances')
        amplitude_width_dist = section.getfloat('amplitude_width_dist')
        amplitude_width_frac = section.getfloat('amplitude_width_frac')
        bottom_temperature = section.getfloat('bottom_temperature')
        surface_temperature = section.getfloat('surface_temperature')
        temperature_difference = section.getfloat('temperature_difference')
        salinity = section.getfloat('salinity')

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

        ds = dsMesh.copy()
        yCell = ds.yCell

        ds['bottomDepth'] = bottom_depth * xarray.ones_like(yCell)
        ds['ssh'] = xarray.zeros_like(yCell)

        init_vertical_coord(config, ds)

        yMin = yCell.min().values
        yMax = yCell.max().values

        yMid = 0.5*(yMin + yMax)

        if use_distances:
            perturbation_width = amplitude_width_dist
        else:
            perturbation_width = (yMax - yMin) * amplitude_width_frac

        # Set stratified temperature
        temp_vert = (bottom_temperature
                     + (surface_temperature - bottom_temperature) *
                       ((ds.refZMid + bottom_depth) / bottom_depth))

        depth_frac = xarray.zeros_like(temp_vert)
        refBottomDepth = ds['refBottomDepth']
        for k in range(1, vert_levels):
            depth_frac[k] = refBottomDepth[k-1] / refBottomDepth[vert_levels-1]

        # If cell is in the southern half, outside the sin width, subtract
        # temperature difference
        frac = xarray.where(numpy.abs(yCell - yMid) < perturbation_width,
                            numpy.cos(0.5 * numpy.pi * (yCell - yMid) /
                                      perturbation_width) *
                            numpy.sin(numpy.pi * depth_frac),
                            0.)

        temperature = temp_vert - temperature_difference * frac
        temperature = temperature.transpose('nCells', 'nVertLevels')
        temperature = temperature.expand_dims(dim='Time', axis=0)

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature
        ds['salinity'] = salinity * xarray.ones_like(temperature)
        ds['normalVelocity'] = normalVelocity

        write_netcdf(ds, 'ocean.nc')
