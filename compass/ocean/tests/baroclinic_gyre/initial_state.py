import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating the initial condition for the baroclinic gyre
    test cases

    Attributes
    ----------
    resolution : float
        The resolution of the test case (m)
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The resolution of the test case (m)
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.add_input_file(
            filename='culled_mesh.nc',
            target='../cull_mesh/culled_mesh.nc')
        self.add_input_file(
            filename='culled_graph.info',
            target='../cull_mesh/culled_graph.info')

        self.add_output_file('initial_state.nc')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config

        dsMesh = xr.open_dataset('culled_mesh.nc')

        ds = _write_initial_state(config, dsMesh)
        print('bottomDepth0', ds.refBottomDepth[0])
        _write_forcing(config, ds.latCell, ds.refBottomDepth)


def _write_initial_state(config, dsMesh):

    ds = dsMesh.copy()

    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')
    ds['bottomDepth'] = bottom_depth * xr.ones_like(ds.xCell)
    ds['ssh'] = xr.zeros_like(ds.xCell)

    init_vertical_coord(config, ds)

    # setting the initial conditions
    section = config['baroclinic_gyre']
    initial_salinity = section.getfloat('initial_salinity')
    temp_top = section.getfloat('initial_temp_top')
    temp_bot = section.getfloat('initial_temp_bot')

    print(f'zmid: {ds.zMid[0,0,:].values}')
    cc = 0.049  # 0.049 value from optimization on 1800m
    aa = temp_top / np.log(cc)
    bb = (cc ** (temp_bot / temp_top) - cc)
    temperature = aa * np.log(bb * -1.0 * ds.zMid / bottom_depth + cc)
    print(f'Tinit: {temperature[0,0,:].values}')
    val_bot = aa * np.log(bb + cc)
    val_top = aa * np.log(cc)
    print(f'analytical bottom T: {val_bot} at '
          f'depth : {bottom_depth}')
    print(f'analytical surface T: {val_top} at '
          f'depth = 0')

#    temperature = (-11. * np.log(0.0414 *
#                   (-1. * ds.zMid + 100.3)) + 48.8)
#    temperature = temperature.transpose('Time', 'nCells', 'nVertLevels')
    print(f'bottom layer T: {temperature[0, 0, -1]} and '
          f'surface layer T: {temperature[0, 0, 0]}')
    salinity = initial_salinity * xr.ones_like(temperature)

    normalVelocity = xr.zeros_like(ds.xEdge)
    normalVelocity, _ = xr.broadcast(normalVelocity, ds.refBottomDepth)
    normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
    normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

    ds['temperature'] = temperature
    ds['salinity'] = salinity
    ds['normalVelocity'] = normalVelocity

    omega = 2. * np.pi / constants['SHR_CONST_SDAY']
    ds['fCell'] = 2. * omega * np.sin(ds.latCell)
    ds['fEdge'] = 2. * omega * np.sin(ds.latEdge)
    ds['fVertex'] = 2. * omega * np.sin(ds.latVertex)

    write_netcdf(ds, 'initial_state.nc')
    return ds


def _write_forcing(config, lat, refBottomDepth):
    section = config['baroclinic_gyre']
    latMin = section.getfloat('lat_min')
    latMax = section.getfloat('lat_max')
    tauMax = section.getfloat('wind_stress_max')
    tempMin = section.getfloat('restoring_temp_min')
    tempMax = section.getfloat('restoring_temp_max')
    restoring_temp_timescale = section.getfloat('restoring_temp_timescale')
    initial_salinity = section.getfloat('initial_salinity')
    lat = np.rad2deg(lat)
    # set wind stress
    windStressZonal = - tauMax * np.cos(2 * np.pi * (lat - latMin) /
                                        (latMax - latMin))

    windStressZonal = windStressZonal.expand_dims(dim='Time', axis=0)

    windStressMeridional = xr.zeros_like(windStressZonal)

    # surface restoring
    temperatureSurfaceRestoringValue = \
        (tempMax - tempMin) * (latMax - lat) / (latMax - latMin) + tempMin
    temperatureSurfaceRestoringValue = \
        temperatureSurfaceRestoringValue.expand_dims(dim='Time', axis=0)

    temperaturePistonVelocity = \
        (refBottomDepth[0] * xr.ones_like(temperatureSurfaceRestoringValue) /
         (restoring_temp_timescale * 24. * 3600.))

    salinitySurfaceRestoringValue = \
        initial_salinity * xr.ones_like(temperatureSurfaceRestoringValue)
    salinityPistonVelocity = xr.zeros_like(temperaturePistonVelocity)

    dsForcing = xr.Dataset()
    dsForcing['windStressZonal'] = windStressZonal
    dsForcing['windStressMeridional'] = windStressMeridional
    dsForcing['temperaturePistonVelocity'] = temperaturePistonVelocity
    dsForcing['salinityPistonVelocity'] = salinityPistonVelocity
    dsForcing['temperatureSurfaceRestoringValue'] = \
        temperatureSurfaceRestoringValue
    dsForcing['salinitySurfaceRestoringValue'] = salinitySurfaceRestoringValue

    write_netcdf(dsForcing, 'forcing.nc')
