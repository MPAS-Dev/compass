from compass.step import Step
from mpas_tools.mesh.interpolation import interp_bilin
from mpas_tools.logging import check_call

import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import subprocess
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class InterpolateAtmForcing(Step):
    """
    A step for interpolating the atmospheric wind velocities
    and pressure onto the MPAS-Ocean mesh to be used as time
    varying forcing in the forward run

    Attributes
    ----------
    plot : bool
        Whether to produce plots of interpolated atmospheric data

    plot_interval : int
        Number of time snaps between plots

    wind_file : str
        Name of file for wind velocity data

    pres_file : str
        Name of file for pressure data

    forcing_file : str
        Output file with interpolated atmospheric data

    self.grid_file : str
        Name of mesh file

    """
    def __init__(self, test_case, mesh, storm):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.init.Init
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        storm : str
            The name of the storm to setup
        """
        super().__init__(test_case=test_case, name='interpolate',
                         ntasks=1, min_tasks=1, openmp_threads=1)

        self.plot = True
        self.plot_interval = 100

        self.wind_file = 'wnd10m.nc'
        self.pres_file = 'prmsl.nc'
        self.forcing_file = 'atmospheric_forcing.nc'
        self.grid_file = 'mesh.nc'

        self.add_input_file(
            filename=self.wind_file,
            target=f'{storm}_wnd10m.nc',
            database='initial_condition_database')

        self.add_input_file(
            filename=self.pres_file,
            target=f'{storm}_prmsl.nc',
            database='initial_condition_database')

        mesh_path = mesh.mesh_step.path

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
        lon_data = data_nc.variables['lon'][:]
        lon_data = np.append(lon_data, 360.0)
        lat_data = np.flipud(data_nc.variables['lat'][:])
        lat_data = np.append(lat_data, 180.0-lat_data[-1])
        time = data_nc.variables['time'][:]
        nsnaps = time.size
        nlon = lon_data.size
        nlat = lat_data.size
        data = np.zeros((nsnaps, nlat, nlon))

        # Get grid from grid file
        lon_grid = grid_nc.variables['lonCell'][:]*180.0/np.pi
        lat_grid = grid_nc.variables['latCell'][:]*180.0/np.pi
        ncells = lon_grid.size
        interp_data = np.zeros((nsnaps, ncells))

        # Interpolate timesnaps
        for i, t in enumerate(time):
            print(f'Interpolating {var}: {i}')

            # Get data to interpolate
            data[i, 0:-1, 0:-1] = np.flipud(data_nc.variables[var][i, :, :])
            data[i, -1, :] = data[i, -2, ::-1]
            data[i, :, -1] = data[i, :, 0]

            interp_data[i, :] = interp_bilin(lon_data, lat_data,
                                             data[i, :, :],
                                             lon_grid, lat_grid)

        # Deal with time
        ref_date = data_nc.variables['time'].getncattr('units')
        ref_date = ref_date.replace('hours since ', '').replace('.0 +0:00', '')
        ref_date = datetime.datetime.strptime(ref_date, '%Y-%m-%d %H:%M:%S')
        xtime = []
        for t in time:
            date = ref_date + datetime.timedelta(hours=np.float64(t))
            xtime.append(date.strftime('%Y-%m-%d_%H:%M:%S'+45*' '))
        xtime = np.array(xtime, 'S64')

        return (lon_grid, lat_grid, interp_data), \
            (lon_data, lat_data, data), \
            xtime

    def plot_interp_data(self, orig_data, interp_data,
                         var_label, var_abrev, time, i):
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

        data = orig_data[2]
        interp = interp_data[2]

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
        ax0.set_title('data '+time.strip().decode())
        cbar = fig.colorbar(cf, ax=ax0)
        cbar.set_label(var_label)

        # Plot interpolated data
        ax1 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        levels = np.linspace(np.amin(interp), np.amax(interp), 100)
        cf = ax1.tricontourf(lon_grid, lat_grid, interp, levels=levels,
                             transform=ccrs.PlateCarree())
        ax1.set_extent([0, 359.9, -90, 90], crs=ccrs.PlateCarree())
        ax1.add_feature(cfeature.LAND, zorder=100)
        ax1.add_feature(cfeature.LAKES, alpha=0.5, zorder=101)
        ax1.add_feature(cfeature.COASTLINE, zorder=101)
        ax1.set_title('interpolated data '+time.strip().decode())
        cbar = fig.colorbar(cf, ax=ax1)
        cbar.set_label(var_label)

        # Save figure
        fig.tight_layout()
        fig.savefig(var_abrev+'_'+str(i).zfill(4)+'.png',
                    box_inches='tight')
        plt.close()

    def write_to_file(self, filename, data, var, xtime):
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
            ncells = data.shape[1]
            nsnaps = data.shape[0]

            # Declare dimensions
            data_nc.createDimension('nCells', ncells)
            data_nc.createDimension('StrLen', 64)
            data_nc.createDimension('Time', None)

            # Create time variable
            time = data_nc.createVariable('xtime', 'S1', ('Time', 'StrLen'))
            time[:, :] = netCDF4.stringtochar(xtime)

        # Set variables
        data_var = data_nc.createVariable(var, np.float64, ('Time', 'nCells'))
        data_var[:, :] = data[:, :]
        data_nc.close()

    def run(self):
        """
        Run this step of the test case
        """
        if os.path.isfile(self.forcing_file):
            check_call(['rm', self.forcing_file], logger=self.logger)

        # Interpolation of u and v velocities
        u_interp, u_data, xtime = self.interpolate_data_to_grid(
            self.grid_file,
            self.wind_file,
            'U_GRD_L103')
        self.write_to_file(self.forcing_file, u_interp[2],
                           'windSpeedU', xtime)

        v_interp,  v_data, xtime = self.interpolate_data_to_grid(
            self.grid_file,
            self.wind_file,
            'V_GRD_L103')
        self.write_to_file(self.forcing_file, v_interp[2],
                           'windSpeedV', xtime)

        # Plot wind velocity
        for i in range(len(xtime)):
            if i % self.plot_interval == 0:
                vel_interp = (u_interp[0], u_interp[1], u_interp[2][i, :])
                vel_interp[2][:] = np.sqrt(np.square(u_interp[2][i, :]) +
                                           np.square(v_interp[2][i, :]))
                vel_data = (u_data[0], u_data[1], u_data[2][i, :, :])
                vel_data[2][:] = np.sqrt(np.square(u_data[2][i, :, :]) +
                                         np.square(v_data[2][i, :, :]))
                self.plot_interp_data(vel_data, vel_interp,
                                      'velocity magnitude', 'vel',
                                      xtime[i], i)

        # Interpolation of atmospheric pressure
        p_interp, p_data, xtime = self.interpolate_data_to_grid(
            self.grid_file,
            self.pres_file,
            'PRMSL_L101')
        self.write_to_file(self.forcing_file, p_interp[2],
                           'atmosPressure', xtime)

        # Plot atmopheric pressure
        for i in range(len(xtime)):
            if i % self.plot_interval == 0:
                press_data = (p_data[0], p_data[1], p_data[2][i, :])
                press_interp = (p_interp[0], p_interp[1], p_interp[2][i, :])
                self.plot_interp_data(press_data,  press_interp,
                                      'atmospheric pressure', 'pres',
                                      xtime[i], i)
