import xarray
import numpy

from compass.ocean.tests.turbulence_closure.boundary_values import add_boundary_arrays
from random import random, seed

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull

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
        surface_salinity = section.getfloat('surface_salinity')
        mixed_layer_salinity_gradient = section.getfloat('mixed_layer_salinity_gradient')
        mixed_layer_temperature_gradient = section.getfloat('mixed_layer_temperature_gradient')
        mixed_layer_depth_salinity = section.getfloat('mixed_layer_depth_salinity')
        mixed_layer_depth_temperature = section.getfloat('mixed_layer_depth_temperature')
        interior_salinity_gradient = section.getfloat('interior_salinity_gradient')
        interior_temperature_gradient = section.getfloat('interior_temperature_gradient')
        surface_heat_flux = section.getfloat('surface_heat_flux')
        surface_freshwater_flux = section.getfloat('surface_freshwater_flux')
        wind_stress_zonal = section.getfloat('wind_stress_zonal')
        wind_stress_meridional = section.getfloat('wind_stress_meridional')
        coriolis_parameter = section.getfloat('coriolis_parameter')
        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=x_nonperiodic,
                                      nonperiodic_y=y_nonperiodic)

        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        ds = dsMesh.copy()
        dsForcing = dsMesh.copy()
        xCell = ds.xCell
        yCell = ds.yCell

        ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
        ds['ssh'] = xarray.zeros_like(xCell)

        init_vertical_coord(config, ds)

        dsForcing['windStressZonal'] = wind_stress_zonal * xarray.ones_like(xCell)
        dsForcing['windStressMeridional'] = wind_stress_meridional * xarray.ones_like(xCell)
        dsForcing['sensibleHeatFlux'] = surface_heat_flux * xarray.ones_like(xCell)
        dsForcing['rainFlux'] = surface_freshwater_flux * xarray.ones_like(xCell)

        write_netcdf(dsForcing, 'init_mode_forcing_data.nc')

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')

        temperature = xarray.zeros_like(ds.layerThickness)

        salinity = xarray.zeros_like(ds.layerThickness)

        surf_indices = numpy.where(ds.refZMid.values >= -mixed_layer_depth_temperature)[0]

        if len(surf_indices) > 0:
            temperature[0,:,surf_indices] = surface_temperature + mixed_layer_temperature_gradient* \
			                                ds.zMid[0,:,surf_indices].values

        int_indices = numpy.where(ds.refZMid.values < -mixed_layer_depth_temperature)[0]

        if len(int_indices) > 0:
            temperature[0,:,int_indices] = surface_temperature - mixed_layer_temperature_gradient* \
													 mixed_layer_depth_temperature + interior_temperature_gradient* \
													 (ds.zMid[0,:,int_indices] + \
													  mixed_layer_depth_temperature)

        temperature[0,:,0] += disturbance_amplitude*2*(numpy.random.rand(len(ds.xCell.values)) - 0.5)

        surf_indices = numpy.where(ds.refZMid.values >= -mixed_layer_depth_salinity)[0]
        if len(surf_indices) > 0:
            salinity[0,:,surf_indices] = surface_salinity - mixed_layer_salinity_gradient * \
			                               	ds.zMid[0,:,surf_indices]

        int_indices = numpy.where(ds.refZMid.values < -mixed_layer_depth_salinity)[0]
        if len(int_indices) > 0:
            salinity[0,:,int_indices] = surface_salinity - mixed_layer_salinity_gradient* \
									  				  mixed_layer_depth_salinity + interior_salinity_gradient* \
													  (ds.zMid[0,:,int_indices] + \
													   mixed_layer_depth_salinity)

        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['normalVelocity'] = normalVelocity
        ds['temperature'] = temperature
        ds['salinity'] = salinity
        ds['fCell'] = coriolis_parameter * xarray.ones_like(xCell)
        ds['fEdge'] = coriolis_parameter * xarray.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xarray.ones_like(ds.xVertex)

        ds = add_boundary_arrays(ds, x_nonperiodic, y_nonperiodic)

        write_netcdf(ds, 'ocean.nc')

