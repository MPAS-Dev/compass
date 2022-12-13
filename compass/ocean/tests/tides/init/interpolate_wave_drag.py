from compass.step import Step
from mpas_tools.mesh.interpolation import interp_bilin

import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class InterpolateWaveDrag(Step):
    """
    A step for interpolating the topographic wave drag data
    onto the MPAS-Ocean mesh

    Attributes
    ----------
    plot : bool
        Whether to produce plots of interpolated atmospheric data

    rinv_file: str
        Name of file for rinv data

    wave_drag_file : str
        Output file with interpolated rinv data

    grid_file : str
        Name of mesh file

    """
    def __init__(self, test_case, mesh):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.tides.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.tides.mesh.Mesh
            The test case that creates the mesh used by this test case
        """
        super().__init__(test_case=test_case, name='interpolate',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.plot = True

        self.rinv_file = 'jsl_lim24_inv_hrs.nc'
        self.wave_drag_file = 'topographic_wave_drag.nc'
        self.grid_file = 'mesh.nc'

        self.add_input_file(
            filename=self.rinv_file,
            target=self.rinv_file,
            database='initial_condition_database')

        mesh_path = mesh.steps['cull_mesh'].path

        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        self.add_output_file(filename=self.wave_drag_file)

    def interpolate_data_to_grid(self, grid_file, data_file):
        """
        Interpolate time snaps of gridded data field to MPAS mesh
        """

        # Open files
        data_nc = netCDF4.Dataset(data_file, 'r')
        grid_nc = netCDF4.Dataset(grid_file, 'r')

        # Get grid from data file
        lon_data = data_nc.variables['Longitude'][:]
        lon_data = np.append(lon_data, 180.0)
        lon_data = np.insert(lon_data, 0, -180.0)
        lat_data = data_nc.variables['Latitude'][:]

        data = data_nc.variables['rinv'][:]
        data = np.squeeze(data)
        data = np.insert(data, 0, data[:, 0], axis=1)
        data = np.append(data, data[:, -1][np.newaxis].T, axis=1)
        data = 1.0/data[:, :]

        # Get grid from grid file
        lon_grid = np.mod(grid_nc.variables['lonEdge'][:] + np.pi,
                          2.0*np.pi) - np.pi
        lon_grid = lon_grid*180.0/np.pi
        lat_grid = grid_nc.variables['latEdge'][:]*180.0/np.pi

        # Interpolate
        interp_data = interp_bilin(lon_data, lat_data,
                                   data,
                                   lon_grid, lat_grid)

        return (lon_grid, lat_grid, interp_data), \
            (lon_data, lat_data, data)

    def plot_interp_data(self, orig_data, interp_data):
        """
        Plot original gridded data and interpolated fields
        """

        plt.switch_backend('agg')

        if not self.plot:
            return

        lon_data = orig_data[0]
        lat_data = orig_data[1]
        lon_grid = interp_data[0]
        lat_grid = interp_data[1]

        data = orig_data[2][:, :]
        interp = interp_data[2][:]

        # Plot data
        fig = plt.figure()
        levels = np.linspace(np.amin(data), np.amax(data), 100)
        ax0 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        cf = ax0.contourf(lon_data, lat_data, data, levels=levels,
                          transform=ccrs.PlateCarree())
        ax0.set_extent([0, 359.9, -90, 90], crs=ccrs.PlateCarree())
        ax0.add_feature(cfeature.LAND, zorder=100)
        ax0.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
        ax0.add_feature(cfeature.COASTLINE, zorder=101)
        ax0.set_title('data')
        cbar = fig.colorbar(cf, ax=ax0)
        cbar.set_label('rinv')

        # Plot interpolated data
        ax1 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        levels = np.linspace(np.amin(interp), np.amax(interp), 100)
        cf = ax1.tricontourf(lon_grid, lat_grid, interp, levels=levels,
                             transform=ccrs.PlateCarree())
        ax1.set_extent([0, 359.9, -90, 90], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, zorder=100)
        ax1.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
        ax1.add_feature(cfeature.COASTLINE, zorder=101)
        ax1.set_title('interpolated data')
        cbar = fig.colorbar(cf, ax=ax1)
        cbar.set_label('rinv')

        # Save figure
        fig.tight_layout()
        fig.savefig('rinv.png',
                    bbox_inches='tight')
        plt.close()

    def write_to_file(self, data):
        """
        Write data to netCDF file
        """

        data_nc = netCDF4.Dataset(self.wave_drag_file, 'w',
                                  format='NETCDF3_64BIT_OFFSET')

        # Find dimesions
        nEdges = data.shape[0]

        # Declare dimensions
        data_nc.createDimension('nEdges', nEdges)

        # Set variables
        data_var = data_nc.createVariable('topographic_wave_drag',
                                          np.float64,
                                          ('nEdges'))
        data_var[:] = data[:]
        data_nc.close()

    def run(self):
        """
        Run this step of the test case
        """

        # Interpolation of rinv
        interp, data = self.interpolate_data_to_grid(
            self.grid_file,
            self.rinv_file)

        self.write_to_file(interp[2])

        # Plot rinv field
        self.plot_interp_data(data, interp)
