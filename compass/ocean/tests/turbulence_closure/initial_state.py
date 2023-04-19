import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.ocean.tests.turbulence_closure.boundary_values import (
    add_boundary_arrays,
)
from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for turbulence closure
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

        for file in ['base_mesh.nc', 'culled_graph.info', 'culled_mesh.nc',
                     'init_mode_forcing_data.nc', 'ocean.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['turbulence_closure']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        x_nonperiodic = section.getboolean('x_nonperiodic')
        y_nonperiodic = section.getboolean('y_nonperiodic')
        disturbance_amplitude = section.getfloat('disturbance_amplitude')
        surface_temperature = section.getfloat('surface_temperature')
        interior_temperature_shape = section.get('interior_temperature_shape')
        interior_salinity_shape = section.get('interior_salinity_shape')
        surface_salinity = section.getfloat('surface_salinity')
        mixed_layer_salinity_gradient = \
            section.getfloat('mixed_layer_salinity_gradient')
        mixed_layer_temperature_gradient = \
            section.getfloat('mixed_layer_temperature_gradient')
        mixed_layer_depth_salinity = \
            section.getfloat('mixed_layer_depth_salinity')
        mixed_layer_depth_temperature = \
            section.getfloat('mixed_layer_depth_temperature')
        interior_salinity_gradient = \
            section.getfloat('interior_salinity_gradient')
        interior_temperature_gradient = \
            section.getfloat('interior_temperature_gradient')
        interior_temperature_c0 = section.getfloat('interior_temperature_c0')
        interior_temperature_c1 = section.getfloat('interior_temperature_c1')
        interior_temperature_c2 = section.getfloat('interior_temperature_c2')
        interior_temperature_c3 = section.getfloat('interior_temperature_c3')
        interior_temperature_c4 = section.getfloat('interior_temperature_c4')
        interior_temperature_c5 = section.getfloat('interior_temperature_c4')
        interior_salinity_c0 = section.getfloat('interior_salinity_c0')
        interior_salinity_c1 = section.getfloat('interior_salinity_c1')
        interior_salinity_c2 = section.getfloat('interior_salinity_c2')
        interior_salinity_c3 = section.getfloat('interior_salinity_c3')
        interior_salinity_c4 = section.getfloat('interior_salinity_c4')
        interior_salinity_c5 = section.getfloat('interior_salinity_c4')
        surface_heat_flux = section.getfloat('surface_heat_flux')
        surface_freshwater_flux = section.getfloat('surface_freshwater_flux')
        wind_stress_zonal = section.getfloat('wind_stress_zonal')
        wind_stress_meridional = section.getfloat('wind_stress_meridional')
        coriolis_parameter = section.getfloat('coriolis_parameter')
        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=x_nonperiodic,
                                      nonperiodic_y=y_nonperiodic)

        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = dsMesh.copy()
        dsForcing = dsMesh.copy()
        xCell = ds.xCell

        ds['bottomDepth'] = bottom_depth * xr.ones_like(xCell)
        ds['ssh'] = xr.zeros_like(xCell)

        init_vertical_coord(config, ds)

        dsForcing['windStressZonal'] = wind_stress_zonal * xr.ones_like(xCell)
        dsForcing['windStressMeridional'] = \
            wind_stress_meridional * xr.ones_like(xCell)
        dsForcing['sensibleHeatFlux'] = surface_heat_flux * xr.ones_like(xCell)
        dsForcing['rainFlux'] = surface_freshwater_flux * xr.ones_like(xCell)

        write_netcdf(dsForcing, 'init_mode_forcing_data.nc')

        normalVelocity = xr.zeros_like(ds.xEdge)
        normalVelocity, _ = xr.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')

        temperature = xr.zeros_like(ds.layerThickness)

        salinity = xr.zeros_like(ds.layerThickness)

        zMid = ds.refZMid.values

        temperature_at_mld = (surface_temperature -
                              mixed_layer_temperature_gradient *
                              mixed_layer_depth_temperature)
        if interior_temperature_shape == 'linear':
            initial_temperature = np.where(
                zMid >= -mixed_layer_depth_temperature,
                surface_temperature + mixed_layer_temperature_gradient * zMid,
                temperature_at_mld + interior_temperature_gradient * zMid)
        elif interior_temperature_shape == 'polynomial':
            initial_temperature = np.where(
                zMid >= -mixed_layer_depth_temperature,
                surface_temperature + mixed_layer_temperature_gradient * zMid,
                interior_temperature_c0 +
                interior_temperature_c1 * zMid +
                interior_temperature_c2 * zMid**2. +
                interior_temperature_c3 * zMid**3. +
                interior_temperature_c4 * zMid**4. +
                interior_temperature_c5 * zMid**5.)
        else:
            print('interior_temperature_shape is not supported')

        salinity_at_mld = (surface_salinity -
                           mixed_layer_salinity_gradient *
                           mixed_layer_depth_salinity)
        if interior_salinity_shape == 'linear':
            initial_salinity = np.where(
                zMid >= -mixed_layer_depth_salinity,
                surface_salinity + mixed_layer_salinity_gradient * zMid,
                salinity_at_mld + interior_salinity_gradient * zMid)
        elif interior_salinity_shape == 'polynomial':
            initial_salinity = np.where(
                zMid >= -mixed_layer_depth_salinity,
                surface_salinity + mixed_layer_salinity_gradient * zMid,
                interior_salinity_c0 +
                interior_salinity_c1 * zMid +
                interior_salinity_c2 * zMid**2. +
                interior_salinity_c3 * zMid**3. +
                interior_salinity_c4 * zMid**4. +
                interior_salinity_c5 * zMid**5.)
        else:
            print('interior_salinity_shape is not supported')

        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)
        ds['normalVelocity'] = normalVelocity

        temperature[0, :, :] = initial_temperature
        temperature[0, :, 0] += disturbance_amplitude * 2. * \
            (np.random.rand(len(ds.xCell.values)) - 0.5)
        salinity[0, :, :] = initial_salinity
        ds['temperature'] = temperature
        ds['salinity'] = salinity
        ds['fCell'] = coriolis_parameter * xr.ones_like(xCell)
        ds['fEdge'] = coriolis_parameter * xr.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xr.ones_like(ds.xVertex)

        ds = add_boundary_arrays(ds, x_nonperiodic, y_nonperiodic)

        write_netcdf(ds, 'ocean.nc')

        plt.figure(dpi=100)
        plt.plot(initial_temperature, zMid, 'k-')
        plt.xlabel('PT (C)')
        plt.ylabel('z (m)')
        plt.savefig('pt_depth_t0h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(dpi=100)
        plt.plot(initial_salinity, zMid, 'k-')
        plt.xlabel('SA (g/kg)')
        plt.ylabel('z (m)')
        plt.savefig('sa_depth_t0h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()
