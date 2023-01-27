import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class Init(Step):
    """
    A step for an initial condition for for the cosine bell test case
    """
    def __init__(self, test_case, mesh_name):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.cosine_bell.CosineBell  # noqa: E501
            The test case this step belongs to

        mesh_name : str
            The name of the mesh
        """

        super().__init__(test_case=test_case,
                         name=f'{mesh_name}_init',
                         subdir=f'{mesh_name}/init',
                         ntasks=36, min_tasks=1)

        self.add_input_file(filename='mesh.nc', target='../mesh/mesh.nc')

        self.add_input_file(filename='graph.info', target='../mesh/graph.info')

        self.add_output_file(filename='initial_state.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config

        section = config['cosine_bell']
        temperature = section.getfloat('temperature')
        salinity = section.getfloat('salinity')
        lat_center = section.getfloat('lat_center')
        lon_center = section.getfloat('lon_center')
        lon_center = section.getfloat('lon_center')
        radius = section.getfloat('radius')
        psi0 = section.getfloat('psi0')
        vel_pd = section.getfloat('vel_pd')

        section = config['vertical_grid']
        bottom_depth = section.getfloat('bottom_depth')

        dsMesh = xr.open_dataset('mesh.nc')
        angleEdge = dsMesh.angleEdge
        latCell = dsMesh.latCell
        latEdge = dsMesh.latEdge
        lonCell = dsMesh.lonCell
        sphere_radius = dsMesh.sphere_radius

        ds = dsMesh.copy()

        ds['bottomDepth'] = bottom_depth * xr.ones_like(latCell)
        ds['ssh'] = xr.zeros_like(latCell)

        init_vertical_coord(config, ds)

        temperature_array = temperature * xr.ones_like(ds.latCell)
        temperature_array, _ = xr.broadcast(temperature_array, ds.refZMid)
        ds['temperature'] = temperature_array.expand_dims(dim='Time', axis=0)
        ds['salinity'] = salinity * xr.ones_like(ds.temperature)

        distance_from_center = sphere_radius * np.arccos(
            np.sin(lat_center) * np.sin(latCell) +
            np.cos(lat_center) * np.cos(latCell) *
            np.cos(lonCell - lon_center))
        bell_value = psi0 / 2.0 * (
            1.0 + np.cos(np.pi *
                         np.divide(distance_from_center, radius)))
        debug_tracers = xr.where(distance_from_center < radius,
                                 bell_value,
                                 0.0)
        debug_tracers_array, _ = xr.broadcast(debug_tracers, ds.refZMid)
        ds['tracer1'] = debug_tracers_array.expand_dims(dim='Time', axis=0)
        ds['tracer2'] = ds.tracer1
        ds['tracer3'] = ds.tracer1

        # Initialize velocity
        seconds_per_day = 86400.0
        velocity = (2.0 * np.pi * np.cos(angleEdge) * sphere_radius *
                    np.cos(latEdge) / (seconds_per_day * vel_pd))
        velocity_array, _ = xr.broadcast(velocity, ds.refZMid)
        ds['normalVelocity'] = velocity_array.expand_dims(dim='Time', axis=0)

        ds['fCell'] = xr.zeros_like(ds.xCell)
        ds['fEdge'] = xr.zeros_like(ds.xEdge)
        ds['fVertex'] = xr.zeros_like(ds.xVertex)

        write_netcdf(ds, 'initial_state.nc')
