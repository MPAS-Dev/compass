import xarray
import numpy

from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for SOMA test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_surface_restoring : bool
        Whether surface restoring is included in the simulation
    """

    def __init__(self, test_case, resolution, with_surface_restoring):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        with_surface_restoring : bool
            Whether surface restoring is included in the simulation
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.with_surface_restoring = with_surface_restoring

        mesh_filenames = {'32km': 'SOMA_32km_grid.161202.nc',
                          '16km': 'SOMA_16km_grid.161202.nc',
                          '8km': 'SOMA_8km_grid.161202.nc',
                          '4km': 'SOMA_4km_grid.161202.nc'}
        if resolution not in mesh_filenames:
            raise ValueError(f'Unexpected SOMA resolution: {resolution}')

        self.add_input_file(filename='base_mesh.nc',
                            target=mesh_filenames[resolution],
                            database='mesh_database')

        for file in ['initial_state.nc', 'forcing.nc', 'graph.info']:
            self.add_output_file(filename=file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        ds_mesh = xarray.open_dataset('base_mesh.nc')

        _add_bottom_depth(config, ds_mesh, add_cull_cell=True)

        write_netcdf(ds_mesh, 'temp.nc')

        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, graphInfoFileName='graph.info',
                          logger=logger)
        write_netcdf(ds_mesh, 'culled_mesh.nc')

        ds = _write_initial_state(config, ds_mesh)

        _write_forcing(config, ds, self.with_surface_restoring)


def _add_bottom_depth(config, ds, add_cull_cell):
    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

    section = config['soma']
    center_longitude = numpy.deg2rad(section.getfloat('center_longitude'))
    center_latitude = numpy.deg2rad(section.getfloat('center_latitude'))
    domain_width = section.getfloat('domain_width')
    shelf_width = section.getfloat('shelf_width')
    shelf_depth = section.getfloat('shelf_depth')
    phi = section.getfloat('phi')

    sphere_radius = ds.attrs['sphere_radius']

    # make sure -pi <= dlon < pi
    dlon = numpy.mod(numpy.abs(ds.lonCell - center_longitude) + numpy.pi,
                     2.*numpy.pi) - numpy.pi
    dlat = ds.latCell - center_latitude

    x = dlon * sphere_radius * numpy.cos(ds.latCell)
    y = dlat * sphere_radius
    distance = numpy.sqrt(x**2 + y**2)

    factor = 1.0 - distance**2/domain_width**2

    ds['bottomDepth'] = shelf_depth + (0.5*(bottom_depth-shelf_depth) *
                                       (1.0 + numpy.tanh(factor/phi)))

    if add_cull_cell:
        ds['cullCell'] = xarray.where(factor > shelf_width, 0, 1)


def _write_initial_state(config, ds_mesh):
    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

    section = config['soma']
    ref_density = section.getfloat('ref_density')
    density_difference_linear = section.getfloat('density_difference_linear')
    density_difference = section.getfloat('density_difference')
    thermocline_depth = section.getfloat('thermocline_depth')
    eos_linear_alpha = section.getfloat('eos_linear_alpha')
    surface_temperature = section.getfloat('surface_temperature')
    surface_salinity = section.getfloat('surface_salinity')

    ds = ds_mesh.copy()

    _add_bottom_depth(config, ds, add_cull_cell=False)

    ds['ssh'] = xarray.zeros_like(ds.xCell)

    init_vertical_coord(config, ds)

    zMid = ds.zMid

    distance = (ref_density -
                ((1.0 - density_difference_linear) * density_difference
                 * numpy.tanh(zMid / thermocline_depth)) -
                (density_difference_linear * density_difference
                 * zMid / bottom_depth))

    factor = (ref_density - distance) / eos_linear_alpha

    temperature = surface_temperature + factor

    factor = - zMid / bottom_depth
    salinity = surface_salinity + 2.0*factor

    normalVelocity = xarray.zeros_like(ds.xEdge)
    normalVelocity = normalVelocity.broadcast_like(ds.refBottomDepth)
    normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
    normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

    ds['temperature'] = temperature
    ds['salinity'] = salinity
    ds['normalVelocity'] = normalVelocity

    # Angular rotation rate of the Earth (1/s)
    # from mpas-framework/src/framework/mpas_constants.F
    omega = 7.29212e-5
    ds['fCell'] = 2.0 * omega * numpy.sin(ds.latCell)
    ds['fEdge'] = 2.0 * omega * numpy.sin(ds.latEdge)
    ds['fVertex'] = 2.0 * omega * numpy.sin(ds.latVertex)

    write_netcdf(ds, 'initial_state.nc')
    return ds


def _write_forcing(config, ds, with_surface_restoring):

    ds_forcing = xarray.Dataset()

    section = config['soma']
    center_latitude = numpy.deg2rad(section.getfloat('center_latitude'))
    domain_width = section.getfloat('domain_width')
    surface_temp_restoring_at_center_latitude = \
        section.getfloat('surface_temp_restoring_at_center_latitude')
    # convert to deg C/rad
    surface_temp_restoring_latitude_gradient = \
        (section.getfloat('surface_temp_restoring_latitude_gradient') /
         numpy.deg2rad(1.))
    restoring_temp_piston_vel = section.getfloat('restoring_temp_piston_vel')

    sphere_radius = ds.attrs['sphere_radius']

    lat = ds.latCell

    # set wind stress
    deltay = sphere_radius * (lat - center_latitude)/domain_width
    factor = 1.0 - 0.5*deltay
    windStressZonal = (factor * 0.1 * numpy.exp(-deltay**2)
                       * numpy.cos(numpy.pi*deltay))

    windStressZonal = windStressZonal.expand_dims(dim='Time', axis=0)

    windStressMeridional = xarray.zeros_like(windStressZonal)

    ds_forcing['windStressZonal'] = windStressZonal
    ds_forcing['windStressMeridional'] = windStressMeridional

    if with_surface_restoring:

        # surface restoring
        temperatureSurfaceRestoringValue = \
            (surface_temp_restoring_at_center_latitude +
             surface_temp_restoring_latitude_gradient*(lat - center_latitude))
        temperatureSurfaceRestoringValue = \
            temperatureSurfaceRestoringValue.expand_dims(dim='Time', axis=0)

        temperaturePistonVelocity = \
            restoring_temp_piston_vel * xarray.ones_like(
                temperatureSurfaceRestoringValue)

        salinitySurfaceRestoringValue = \
            34.0 * xarray.ones_like(temperatureSurfaceRestoringValue)
        salinityPistonVelocity = xarray.zeros_like(temperaturePistonVelocity)

        ds_forcing['temperaturePistonVelocity'] = temperaturePistonVelocity
        ds_forcing['salinityPistonVelocity'] = salinityPistonVelocity
        ds_forcing['temperatureSurfaceRestoringValue'] = \
            temperatureSurfaceRestoringValue
        ds_forcing['salinitySurfaceRestoringValue'] = \
            salinitySurfaceRestoringValue

    write_netcdf(ds_forcing, 'forcing.nc')
