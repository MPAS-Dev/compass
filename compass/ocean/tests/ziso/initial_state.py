import os
import xarray
import numpy

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

from compass.testcase import get_step_default
from compass.ocean.vertical import generate_grid
from compass.ocean.vertical.zstar import compute_layer_thickness_and_zmid


def collect(resolution):
    """
    Get a dictionary of step properties

    Parameters
    ----------
    resolution : {'20km'}
        The name of the resolution to run at

    Returns
    -------
    step : dict
        A dictionary of properties of this step
    """
    step = get_step_default(__name__)
    step['resolution'] = resolution

    step['cores'] = 1
    step['min_cores'] = 1
    # Maximum allowed memory and disk usage in MB
    step['max_memory'] = 8000
    step['max_disk'] = 8000

    return step


def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core, configuration and testcase
    """
    step_dir = step['work_dir']

    inputs = []
    outputs = []

    for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                 'ocean.nc', 'forcing.nc']:
        outputs.append(os.path.join(step_dir, file))

    step['inputs'] = inputs
    step['outputs'] = outputs


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()`` function,
        with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
   """
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

    ds = _write_initial_state(config, dsMesh)

    _write_forcing(config, ds.yCell, ds.zMid)


def _write_initial_state(config, dsMesh):
    section = config['ziso']
    reference_coriolis = section.getfloat('reference_coriolis')
    coriolis_gradient = section.getfloat('coriolis_gradient')

    ds = dsMesh.copy()

    interfaces = generate_grid(config=config)
    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

    ds['refBottomDepth'] = ('nVertLevels', interfaces[1:])
    ds['refZMid'] = ('nVertLevels', -0.5 * (interfaces[1:] + interfaces[0:-1]))
    ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

    yCell = ds.yCell

    shelf_depth = section.getfloat('shelf_depth')
    slope_center_position = section.getfloat('slope_center_position')
    slope_half_width = section.getfloat('slope_half_width')

    bottomDepth = shelf_depth + 0.5 * (bottom_depth - shelf_depth) * \
                  (1.0 + numpy.tanh((yCell - slope_center_position) /
                                    slope_half_width))

    refTopDepth = xarray.DataArray(data=interfaces[0:-1], dims=('nVertLevels',))

    cellMask = (refTopDepth < bottomDepth).transpose('nCells', 'nVertLevels')

    maxLevelCell = cellMask.sum(dim='nVertLevels') - 1

    # We want full cells, so deepen bottomDepth to be the bottom of the last
    # valid layer
    bottomDepth = ds.refBottomDepth.isel(nVertLevels=maxLevelCell)

    restingThickness, layerThickness, zMid = compute_layer_thickness_and_zmid(
        cellMask, ds.refBottomDepth, bottomDepth, maxLevelCell)

    layerThickness = layerThickness.expand_dims(dim='Time', axis=0)
    zMid = zMid.expand_dims(dim='Time', axis=0)

    initial_temp_t1 = section.getfloat('initial_temp_t1')
    initial_temp_t2 = section.getfloat('initial_temp_t2')
    initial_temp_h1 = section.getfloat('initial_temp_h1')
    initial_temp_mt = section.getfloat('initial_temp_mt')
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
    ds['layerThickness'] = layerThickness
    ds['restingThickness'] = layerThickness
    ds['zMid'] = zMid
    ds['bottomDepth'] = bottomDepth
    # fortran 1-based indexing
    ds['maxLevelCell'] = maxLevelCell+1
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
    meridional_extent = section.getfloat('meridional_extent')
    front_width = section.getfloat('antarctic_shelf_front_width')
    front_max = section.getfloat('wind_stress_shelf_front_max')
    restoring_sponge_l = section.getfloat('restoring_sponge_l')
    restoring_temp_ze = section.getfloat('restoring_temp_ze')
    restoring_temp_tau = section.getfloat('restoring_temp_tau')

    # set wind stress
    windStressZonal = xarray.where(
        yCell >= y_trans,
        wind_stress_max * numpy.sin(numpy.pi * (yCell - y_trans) /
                                    (meridional_extent - y_trans))**2,
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
    mask = meridional_extent - yCell <= 1.5 * restoring_sponge_l
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
        mask, 34.0 , salinityInteriorRestoringValue)

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
    dsForcing['salinityInteriorRestoringValue'] = salinityInteriorRestoringValue

    write_netcdf(dsForcing, 'forcing.nc')
