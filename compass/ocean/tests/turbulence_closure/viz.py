import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing a cross-section through the domain
    """
    def __init__(self, test_case, resolution, forcing, do_comparison=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        self.add_input_file(filename='mesh.nc',
                            target='../initial_state/culled_mesh.nc')

        if do_comparison:
            if forcing == 'cooling':
                forcing_name = 'c02'
            elif forcing == 'evaporation':
                forcing_name = 'e04'
            else:
                do_comparison = False

        if do_comparison:
            # Compare all cases with 1m resolution PALM output
            suffix = 'g128_l128'
            filename = f'case_{forcing_name}_{suffix}.nc'
            self.add_input_file(filename='palm.nc', target=filename,
                                database='turbulence_closure')

        self.do_comparison = do_comparison

    def run(self):
        """
        Run this step of the test case
        """

        figsize = [6.4, 4.8]
        markersize = 5

        dsInit = xr.open_dataset('../forward/init.nc')
        ds = xr.open_dataset('output.nc')
        if self.do_comparison:
            ds_palm = xr.open_dataset('palm.nc')

        if 'Time' not in ds.dims:
            print('Dataset missing time dimension')
            return

        nt = ds.sizes['Time']  # number of timesteps
        tidx = nt - 1
        seconds_per_day = 24.0 * 3600.0
        if 'daysSinceStartOfSim' in ds.keys():
            time = ds.daysSinceStartOfSim
        else:
            # This routine is not generic but should not be used as
            # daysSinceStartOfSim is included in the streams file
            time0 = 2.0 + (7.0 / 24.0)
            dt = 0.5 / 24.0
            time = np.linspace(time0 + dt, time0 + nt * dt, num=nt)

        if self.do_comparison:
            time_palm = ds_palm.time
            time_palm_day = (time_palm.astype('float64') /
                             (seconds_per_day * 1e9))
            tidx_palm = np.argmin(np.abs(np.subtract(
                time_palm_day.values, time[tidx])))

        ds = ds.isel(Time=tidx)
        ds_palm = ds_palm.isel(time=tidx_palm)

        if 'yEdge' not in ds.keys():
            ds['yEdge'] = dsInit.yEdge
        ds = ds.sortby('yEdge')

        # Get mesh variables
        xCell = dsInit.xCell
        yCell = dsInit.yCell
        zMid = dsInit.refZMid

        # Import cell quantities
        temperature = ds.temperature
        temperature_z = temperature.mean(dim='nCells')
        salinity = ds.salinity
        salinity_z = salinity.mean(dim='nCells')
        w = ds.verticalVelocity
        w_zTop = w[:, 0]

        velocityZonal = ds.velocityZonal
        velocityZonal_z = velocityZonal.mean(dim='nCells')
        velocityMeridional = ds.velocityMeridional
        velocityMeridional_z = velocityMeridional.mean(dim='nCells')
        buoyancyProduction = ds.buoyancyProduction
        buoyancyProduction_z = buoyancyProduction.mean(dim='nCells')
        wpt = ds.temperatureVerticalAdvectionTendency
        wpt_z = wpt.mean(dim='nCells')

        if self.do_comparison:
            alpha_T = ds_palm.alpha_T
            if 'beta_S' in ds_palm.keys():
                beta_S = ds_palm.beta_S
            else:
                beta_S = 7.8e-4
            pt_palm = ds_palm.pt - 273.15
            sa_palm = ds_palm.sa
            u_palm = ds_palm.u
            v_palm = ds_palm.v
            wpt_palm = np.add(ds_palm['w*pt*'].values, ds_palm['w"pt"'].values)
            wsa_palm = np.add(ds_palm['w*sa*'].values, ds_palm['w"sa"'].values)
            temp1 = np.multiply(alpha_T, wpt_palm)
            temp2 = np.multiply(beta_S, wsa_palm)
            buoyancyProduction_palm = np.multiply(9.81,
                                                  np.subtract(temp1, temp2))
            zu_palm = ds_palm.zu
            z_palm = ds_palm.zprho

        # Figures

        plt.figure(figsize=figsize, dpi=100)
        plt.plot(temperature_z.values, zMid, 'k-')
        if self.do_comparison:
            plt.plot(pt_palm.values, z_palm.values, 'b-')
        plt.xlabel('PT (C)')
        plt.ylabel('z (m)')
        plt.savefig(f'pt_depth_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(figsize=figsize, dpi=100)
        plt.plot(wpt_z, zMid, 'k-')
        if self.do_comparison:
            plt.plot(wpt_palm, z_palm.values, 'b-')
        plt.xlabel('wpt (C m/s)')
        plt.ylabel('z (m)')
        plt.savefig(f'wpt_depth_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(figsize=figsize, dpi=100)
        plt.plot(buoyancyProduction_z, zMid, 'k-')
        if self.do_comparison:
            plt.plot(buoyancyProduction_palm, z_palm.values, 'b-')
        plt.xlabel('bouyancy production')
        plt.ylabel('z (m)')
        plt.savefig(f'buoy_depth_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(figsize=figsize, dpi=100)
        plt.plot(salinity_z.values, zMid, 'k-')
        if self.do_comparison:
            plt.plot(sa_palm.values, z_palm.values, 'b-')
        plt.xlabel('SA (g/kg)')
        plt.ylabel('z (m)')
        plt.savefig(f'sa_depth_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(figsize=figsize, dpi=100)
        plt.plot(velocityZonal_z.values, zMid, 'k-')
        plt.plot(velocityMeridional_z.values, zMid, 'k--')
        if self.do_comparison:
            plt.plot(u_palm.values, zu_palm.values, 'b-')
            plt.plot(v_palm.values, zu_palm.values, 'b--')
        plt.xlabel('u,v (m/s)')
        plt.ylabel('z (m)')
        plt.savefig(f'uv_depth_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()

        plt.figure(figsize=figsize, dpi=100)
        cmax = np.max(np.abs(w_zTop.values))
        plt.scatter(np.divide(xCell, 1e3),
                    np.divide(yCell, 1e3),
                    s=markersize, c=w_zTop.values,
                    cmap='cmo.balance', vmin=-1. * cmax, vmax=cmax)
        plt.xlabel('x (km)')
        plt.ylabel('y (km)')
        cbar = plt.colorbar()
        cbar.ax.set_title('w (m/s)')
        plt.savefig(f'w_top_section_t{int(time[tidx]*24.0)}h.png',
                    bbox_inches='tight', dpi=200)
        plt.close()
