import os

import netCDF4
import numpy as np
from mpas_tools.logging import check_call
from scipy import interpolate

from compass.step import Step


class ComputeTopographicWaveDrag(Step):
    """
    A step for computing the topographic wave drag
    forcing term

    Attributes
    ----------
    rinv_file : str
        Name of file for rinv data

    output_file : str
        Output file with topographic wave drag data

    self.grid_file : str
        Name of mesh file

    """
    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane_lts.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case
        """
        super().__init__(test_case=test_case, name='topodrag',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.rinv_file = 'jsl_lim24_inv_hrs.nc'
        self.forcing_file = 'topographic_wave_drag.nc'
        self.grid_file = 'mesh.nc'

        self.add_input_file(
            filename=self.rinv_file,
            target='jsl_lim24_inv_hrs.nc',
            database='initial_condition_database')

        mesh_path = mesh.steps['cull_mesh'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_output_file(filename=self.forcing_file)

    def interpolate_data_to_grid(self, grid_file, data_file, var):
        """
        Interpolate time snaps of gridded data field to MPAS mesh
        """

        # Open files
        data_nc = netCDF4.Dataset(data_file, 'r')
        grid_nc = netCDF4.Dataset(grid_file, 'r')

        # Get grid from data file
        lon_data = data_nc.variables['Longitude'][:]
        lat_data = data_nc.variables['Latitude'][:]
        time = data_nc.variables['MT'][:]
        nsnaps = time.size

        # Get grid from grid file
        lonEdge = grid_nc.variables['lonEdge'][:]
        latEdge = grid_nc.variables['latEdge'][:]
        lon_grid = np.mod(lonEdge + np.pi, 2.0 * np.pi) - np.pi
        lon_grid = lon_grid * 180.0 / np.pi
        lat_grid = latEdge * 180.0 / np.pi

        grid_points = np.column_stack((lon_grid, lat_grid))
        nEdges = lon_grid.size
        interp_data = np.zeros((nsnaps, nEdges))

        # Get data to interpolate
        data = data_nc.variables[var][:]

        # Interpolate data onto new grid
        interpolator = interpolate.RegularGridInterpolator((lon_data,
                                                           lat_data),
                                                           1 / data[0, :, :].T,
                                                           bounds_error=False,
                                                           fill_value=None,
                                                           method='nearest')
        interp_data[0, :] = interpolator(grid_points)

        return interp_data

    def write_to_file(self, filename, data, var):
        """
        Write data to netCDF file
        """

        if os.path.isfile(filename):
            data_nc = netCDF4.Dataset(filename, 'a',
                                      format='NETCDF3_64BIT_OFFSET')
        else:
            data_nc = netCDF4.Dataset(filename, 'w',
                                      format='NETCDF3_64BIT_OFFSET')

            # Find dimesions
            nedges = data.shape[1]

            # Declare dimensions
            data_nc.createDimension('nEdges', nedges)
            data_nc.createDimension('StrLen', 64)

        # Set variables
        data_var = data_nc.createVariable(var, np.float64, ('nEdges'))
        data_var[:] = data[:]
        data_nc.close()

    def run(self):
        """
        Run this step of the test case
        """
        if os.path.isfile(self.forcing_file):
            check_call(['rm', self.forcing_file], logger=self.logger)

        # Interpolation of topographic wave drag
        rinv_interp = self.interpolate_data_to_grid(self.grid_file,
                                                    self.rinv_file, 'rinv')

        self.write_to_file(self.forcing_file, rinv_interp,
                           'topographic_wave_drag')
