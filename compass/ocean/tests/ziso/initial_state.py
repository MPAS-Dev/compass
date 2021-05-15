import xarray
import numpy

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for ZISO test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_frazil : bool
        Whether frazil formation is included in the simulation
    """

    def __init__(self, test_case, resolution, with_frazil):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        with_frazil : bool
            Whether frazil formation is included in the simulation
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.with_frazil = with_frazil

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'ocean.nc', 'forcing.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['ziso']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=True)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = _write_initial_state(config, dsMesh, self.with_frazil)

        _write_forcing(config, ds.yCell, ds.zMid)


def _write_initial_state(config, dsMesh, with_frazil):
    section = config['ziso']
    reference_coriolis = section.getfloat('reference_coriolis')
    coriolis_gradient = section.getfloat('coriolis_gradient')

    ds = dsMesh.copy()

    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

    xCell = ds.xCell
    yCell = ds.yCell

    shelf_depth = section.getfloat('shelf_depth')
    slope_center_position = section.getfloat('slope_center_position')
    slope_half_width = section.getfloat('slope_half_width')

    ds['bottomDepth'] = (shelf_depth + 0.5 * (bottom_depth - shelf_depth) *
                         (1.0 + numpy.tanh((yCell - slope_center_position) /
                                           slope_half_width)))

    ds['ssh'] = xarray.zeros_like(xCell)

    init_vertical_coord(config, ds)

    zMid = ds.zMid

    initial_temp_t1 = section.getfloat('initial_temp_t1')
    initial_temp_t2 = section.getfloat('initial_temp_t2')
    initial_temp_h1 = section.getfloat('initial_temp_h1')
    initial_temp_mt = section.getfloat('initial_temp_mt')
    if with_frazil:
        extent = section.getfloat('meridional_extent')
        frazil_anomaly = section.getfloat('frazil_temperature_anomaly')
        distanceX = extent/4.0 - xCell
        distanceY = extent/2.0 - yCell
        distance = numpy.sqrt(distanceY**2 + distanceX**2)
        scaleFactor = numpy.exp(-distance/extent*20.0)

        mask = zMid > -50.

        frazil_temp = (frazil_anomaly +
                       initial_temp_t2 * numpy.tanh(zMid / initial_temp_h1) +
                       initial_temp_mt * zMid +
                       mask * 1.0*numpy.cos(zMid/50.0 * numpy.pi/2.0))

        temperature = (initial_temp_t1 +
                       initial_temp_t2 * numpy.tanh(zMid / initial_temp_h1) +
                       initial_temp_mt * zMid)

        temperature = ((1.0-scaleFactor) * temperature +
                       scaleFactor * frazil_temp)
        temperature = temperature.transpose('Time', 'nCells', 'nVertLevels')
    else:
        temperature = (initial_temp_t1 +
                       initial_temp_t2 * numpy.tanh(zMid / initial_temp_h1) +
                       initial_temp_mt * zMid)

    salinity = 34.0 * xarray.ones_like(temperature)

    normalVelocity = xarray.zeros_like(ds.xEdge)
    normalVelocity = normalVelocity.broadcast_like(ds.refBottomDepth)
    normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
    normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

    ds['temperature'] = temperature
    ds['salinity'] = salinity
    ds['normalVelocity'] = normalVelocity
    ds['fCell'] = reference_coriolis + yCell * coriolis_gradient
    ds['fEdge'] = reference_coriolis + ds.yEdge * coriolis_gradient
    ds['fVertex'] = reference_coriolis + ds.yVertex * coriolis_gradient

    write_netcdf(ds, 'ocean.nc')
    return ds


def _write_forcing(config, yCell, zMid):
    section = config['ziso']

    extent = section.getfloat('meridional_extent')
    mean_restoring_temp = section.getfloat('mean_restoring_temp')
    restoring_temp_dev_ta = section.getfloat('restoring_temp_dev_ta')
    restoring_temp_dev_tb = section.getfloat('restoring_temp_dev_tb')
    restoring_temp_piston_vel = section.getfloat('restoring_temp_piston_vel')
    y_trans = section.getfloat('wind_transition_position')
    wind_stress_max = section.getfloat('wind_stress_max')
    front_width = section.getfloat('antarctic_shelf_front_width')
    front_max = section.getfloat('wind_stress_shelf_front_max')
    restoring_sponge_l = section.getfloat('restoring_sponge_l')
    restoring_temp_ze = section.getfloat('restoring_temp_ze')
    restoring_temp_tau = section.getfloat('restoring_temp_tau')

    # set wind stress
    windStressZonal = xarray.where(
        yCell >= y_trans,
        wind_stress_max * numpy.sin(numpy.pi * (yCell - y_trans) /
                                    (extent - y_trans))**2,
        front_max * numpy.sin(numpy.pi * (y_trans - yCell) /
                              front_width)**2)

    windStressZonal = xarray.where(yCell >= y_trans - front_width,
                                   windStressZonal, 0.0)

    windStressZonal = windStressZonal.expand_dims(dim='Time', axis=0)

    windStressMeridional = xarray.zeros_like(windStressZonal)

    arg = (yCell - 0.5 * extent) / (0.5 * extent)

    # surface restoring
    temperatureSurfaceRestoringValue = \
        (mean_restoring_temp + restoring_temp_dev_ta * numpy.tanh(2.0*arg) +
         restoring_temp_dev_tb * arg)
    temperatureSurfaceRestoringValue = \
        temperatureSurfaceRestoringValue.expand_dims(dim='Time', axis=0)

    temperaturePistonVelocity = \
        restoring_temp_piston_vel * xarray.ones_like(
            temperatureSurfaceRestoringValue)

    salinitySurfaceRestoringValue = \
        34.0 * xarray.ones_like(temperatureSurfaceRestoringValue)
    salinityPistonVelocity = xarray.zeros_like(temperaturePistonVelocity)

    # set restoring at northern boundary
    mask = extent - yCell <= 1.5 * restoring_sponge_l
    mask = mask.broadcast_like(zMid).transpose('Time', 'nCells', 'nVertLevels')

    # convert from days to inverse seconds
    rate = 1.0 / (restoring_temp_tau*86400.0)

    temperatureInteriorRestoringValue = xarray.where(
        mask, (temperatureSurfaceRestoringValue *
               numpy.exp(zMid/restoring_temp_ze)), 0.)

    temperatureInteriorRestoringRate = xarray.where(
        mask, numpy.exp(-(extent-yCell)/restoring_sponge_l) * rate, 0.)

    salinityInteriorRestoringValue = xarray.where(
        mask, 34.0, 0)

    # set restoring at southern boundary
    mask = yCell <= 2.0 * restoring_sponge_l
    mask = mask.broadcast_like(zMid).transpose('Time', 'nCells', 'nVertLevels')

    temperatureInteriorRestoringValue = xarray.where(
        mask, temperatureSurfaceRestoringValue,
        temperatureInteriorRestoringValue)

    temperatureInteriorRestoringRate = xarray.where(
        mask, numpy.exp(-yCell/restoring_sponge_l) * rate,
        temperatureInteriorRestoringRate)

    salinityInteriorRestoringValue = xarray.where(
        mask, 34.0, salinityInteriorRestoringValue)

    salinityInteriorRestoringRate = \
        xarray.zeros_like(temperatureInteriorRestoringRate)

    dsForcing = xarray.Dataset()
    dsForcing['windStressZonal'] = windStressZonal
    dsForcing['windStressMeridional'] = windStressMeridional
    dsForcing['temperaturePistonVelocity'] = temperaturePistonVelocity
    dsForcing['salinityPistonVelocity'] = salinityPistonVelocity
    dsForcing['temperatureSurfaceRestoringValue'] = \
        temperatureSurfaceRestoringValue
    dsForcing['salinitySurfaceRestoringValue'] = salinitySurfaceRestoringValue
    dsForcing['temperatureInteriorRestoringRate'] = \
        temperatureInteriorRestoringRate
    dsForcing['salinityInteriorRestoringRate'] = salinityInteriorRestoringRate
    dsForcing['temperatureInteriorRestoringValue'] = \
        temperatureInteriorRestoringValue
    dsForcing['salinityInteriorRestoringValue'] = \
        salinityInteriorRestoringValue

    write_netcdf(dsForcing, 'forcing.nc')
