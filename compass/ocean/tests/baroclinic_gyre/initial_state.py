import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating the initial condition for the baroclinic gyre
    test cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case
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

        _write_forcing(config, ds.latCell)


def _write_initial_state(config, dsMesh):
    # section = config['baroclinic_gyre']

    ds = dsMesh.copy()

    bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')
    ds['bottomDepth'] = bottom_depth * xr.ones_like(ds.xCell)
    ds['ssh'] = xr.zeros_like(ds.xCell)

    init_vertical_coord(config, ds)

    # setting the initial conditions
    temperature = (-11. * np.log(0.0414 *
                   (-1. * ds.zMid + 100.3)) + 48.8)
    temperature = temperature.transpose('Time', 'nCells', 'nVertLevels')
    print(f'bottom T: {temperature[0, 200, -1]} and \
            surface : {temperature[0, 200, 0]}')
    salinity = 34.0 * xr.ones_like(temperature)

    normalVelocity = xr.zeros_like(ds.xEdge)
    normalVelocity, _ = xr.broadcast(normalVelocity, ds.refBottomDepth)
    normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
    normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

    ds['temperature'] = temperature
    ds['salinity'] = salinity
    ds['normalVelocity'] = normalVelocity

    omega = 2. * np.pi / 86164.
    ds['fCell'] = 2. * omega * np.sin(ds.latCell)
    ds['fEdge'] = 2. * omega * np.sin(ds.latEdge)
    ds['fVertex'] = 2. * omega * np.sin(ds.latVertex)

    write_netcdf(ds, 'initial_state.nc')
    return ds


def _write_forcing(config, lat):
    section = config['baroclinic_gyre']
    latMin = section.getfloat('lat_min')
    latMax = section.getfloat('lat_max')
    tauMax = section.getfloat('wind_stress_max')
    tempMin = section.getfloat('temp_min')
    tempMax = section.getfloat('temp_max')
    restoring_temp_piston_vel = section.getfloat('restoring_temp_piston_vel')
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
        restoring_temp_piston_vel * xr.ones_like(
            temperatureSurfaceRestoringValue)

    salinitySurfaceRestoringValue = \
        34.0 * xr.ones_like(temperatureSurfaceRestoringValue)
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
