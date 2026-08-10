
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpas_tools.mesh.creation.util import lonlat2xyz
from scipy.spatial import KDTree

from compass.step import Step


class InterpolateManningsN(Step):
    """
    A step for interpolating the Mannings n values used for
    bottom drag

    Attributes
    ----------

    self.grid_file : str
        Name of mesh file

    """
    def __init__(self, test_case, init):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.init.Init
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='interpolate_mannings_n',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.plot = True

        self.data_file = 'mannings_n.nc'
        self.grid_file = 'init.nc'

        self.add_input_file(
            filename=self.data_file,
            target='gstofs_mannings_n.nc',
            database='hurricane')

        self.add_input_file(
            filename=self.grid_file,
            work_dir_target=f'{init.path}/ocean.nc')

    def interpolate_data(self, grid_file, data_file):
        """
        Interpolate time snaps of gridded data field to MPAS mesh
        """

        # Open files
        data_nc = netCDF4.Dataset(data_file, 'r')
        grid_nc = netCDF4.Dataset(grid_file, 'r+')

        # Get grid from data file
        lon_data = data_nc.variables['lon'][:]
        lat_data = data_nc.variables['lat'][:]
        data = data_nc.variables['manning_n'][:]
        npts = lon_data.size

        # Get grid from grid file
        lon_grid = grid_nc.variables['lonCell'][:] * 180.0 / np.pi
        lat_grid = grid_nc.variables['latCell'][:] * 180.0 / np.pi
        lon_grid = np.mod(lon_grid + 180.0, 360.0) - 180.0
        area = grid_nc.variables['areaCell'][:]
        ncells = lon_grid.size
        bottom_drag = np.zeros_like(area)

        # Interpolate using area averaging
        xyz_data = np.zeros((npts, 3))
        xyz_data[:, 0], xyz_data[:, 1], xyz_data[:, 2] = lonlat2xyz(lon_data,
                                                                    lat_data)
        tree = KDTree(xyz_data)

        radius = np.sqrt(area / np.pi)
        x, y, z = lonlat2xyz(lon_grid, lat_grid)
        cells = np.vstack([x, y, z]).T
        idx = tree.query_ball_point(cells, radius)
        d, nearest = tree.query(cells)

        for i in range(ncells):
            if len(idx[i]) > 0:
                bottom_drag[i] = np.mean(data[idx[i]])
            else:
                bottom_drag[i] = data[nearest[i]]

        nc_vars = grid_nc.variables.keys()
        if 'bottomDrag' not in nc_vars:
            grid_nc.createVariable('bottomDrag', 'f8', ('nCells'))

        # Write to mesh file
        grid_nc.variables['bottomDrag'][:] = bottom_drag

        return lon_grid, lat_grid, bottom_drag

    def plot_interp_data(self, lon, lat, var):
        """
        Plot interpolated field
        """

        plt.switch_backend('agg')

        if not self.plot:
            return

        # Plot interpolated data
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        levels = np.linspace(np.amin(var), np.amax(var), 100)
        cf = ax1.tricontourf(lon, lat, var, levels=levels,
                             transform=ccrs.PlateCarree())
        ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, zorder=100)
        ax1.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
        ax1.add_feature(cfeature.COASTLINE, zorder=101)
        ax1.set_title('interpolated manning n')
        cbar = fig.colorbar(cf, ax=ax1)
        cbar.set_label('Mannings n')

        # Save figure
        fig.tight_layout()
        fig.savefig('mannings_n.png',
                    bbox_inches='tight')
        plt.close()

    def run(self):
        """
        Run this step of the test case
        """

        lon, lat, drag = self.interpolate_data(self.grid_file, self.data_file)
        self.plot_interp_data(lon, lat, drag)
