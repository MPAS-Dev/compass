import datetime as dt
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing parabolic bowl results and
    comparing with analytical solution

    Attributes
    ----------
    """
    def __init__(self, test_case, resolutions, use_lts):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        self.resolutions = resolutions
        self.use_lts = use_lts

        for res in resolutions:
            self.add_input_file(filename=f'output_{res}km.nc',
                                target=f'../forward_{res}km/output.nc')

    def run(self):
        """
        Run this step of the test case
        """

        points = self.get_points()
        self.timeseries_plots(points)
        self.inject_exact_solution()
        self.contour_plots(points)
        self.rmse_plots()

    def get_points(self):
        """
        Get the point coordinates for plotting solution timeseries
        """

        points = self.config.get('parabolic_bowl_viz', 'points')
        points = points.replace('[', '').replace(']', '').split(',')
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        points = points * 1000

        return points

    def timeseries_plots(self, points):
        """
        Plot solution timeseries at a given number of points
        for each resolution
        """

        fig, ax = plt.subplots(nrows=len(points), ncols=1)

        for res in self.resolutions:
            ds = xr.open_dataset(f'output_{res}km.nc')

            time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                    for x in ds.xtime.values]
            t = np.asarray([(x - time[0]).total_seconds() for x in time])

            xy = np.vstack((ds.xCell.values, ds.yCell.values)).T
            interp = LinearNDInterpolator(xy, ds.ssh.values.T)

            for i, pt in enumerate(points):

                ssh = interp(pt).T
                ax[i].plot(t / 86400, ssh, label=f'{res}km')

        for i, pt in enumerate(points):
            ssh_exact = self.exact_solution('zeta', pt[0], pt[1], t)
            ax[i].plot(t / 86400, ssh_exact, label='exact')

        for i, pt in enumerate(points):
            ax[i].set_xlabel('t (days)')
            ax[i].set_ylabel('ssh (m)')
            ax[i].set_title(f'Point ({pt[0]/1000}, {pt[1]/1000})')
            if i == len(points) - 1:
                lines, labels = ax[i].get_legend_handles_labels()

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.legend(lines, labels,
                   loc='lower center', ncol=4)
        fig.savefig('points.png')

    def inject_exact_solution(self):
        """
        Save exact solution to output nc file
        """

        for res in self.resolutions:
            ds = xr.open_dataset(f'output_{res}km.nc')

            if 'ssh_exact' and 'layerThickness_exact' not in ds:
                time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                        for x in ds.xtime.values]
                ssh_exact = ds.ssh.copy(deep=True)
                layerThickness_exact = ds.layerThickness.copy(deep=True)
                for i, tstep in enumerate(time):
                    t = (time[i] - time[0]).total_seconds()

                    ssh_exact[i, :] = self.exact_solution(
                        'zeta', ds.xCell.values, ds.yCell.values, t)
                    layerThickness_exact[i, :, 0] = self.exact_solution(
                        'h', ds.xCell.values, ds.yCell.values, t)
                ds['ssh_exact'] = ssh_exact
                ds['layerThickness_exact'] = layerThickness_exact
                ds.ssh_exact.encoding['_FillValue'] = None
                ds.layerThickness_exact.encoding['_FillValue'] = None
                ds.to_netcdf(f'output_{res}km.nc',
                             format="NETCDF3_64BIT_OFFSET", mode='a')
            ds.close()

    def contour_plots(self, points):
        """
        Plot contour plots at a specified output interval for each resolution
        and show where the points used in `points.png` are located.
        """

        sol_min = -2
        sol_max = 2
        clevels = np.linspace(sol_min, sol_max, 50)
        cmap = plt.get_cmap('RdBu')

        ds = xr.open_dataset(f'output_{self.resolutions[0]}km.nc')
        time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                for x in ds.xtime.values]
        ds.close()

        plot_interval = self.config.getint('parabolic_bowl_viz',
                                           'plot_interval')
        for i, tstep in enumerate(time):

            if i % plot_interval != 0:
                continue

            ncols = len(self.resolutions) + 1
            fig, ax = plt.subplots(nrows=1, ncols=ncols,
                                   figsize=(5 * ncols, 5),
                                   constrained_layout=True)

            for j, res in enumerate(self.resolutions):
                ds = xr.open_dataset(f'output_{res}km.nc')
                ax[j].tricontourf(ds.xCell / 1000, ds.yCell / 1000,
                                  ds['ssh'][i, :],
                                  levels=clevels, cmap=cmap,
                                  vmin=sol_min, vmax=sol_max, extend='both')
                ax[j].set_aspect('equal', 'box')
                ax[j].set_title(f'{res}km resolution')
                ax[j].set_xlabel('x (km)')
                ax[j].set_ylabel('y (km)')
                ds.close()

            ds = xr.open_dataset(f'output_{min(self.resolutions)}km.nc')
            cm = ax[ncols - 1].tricontourf(ds.xCell / 1000, ds.yCell / 1000,
                                           ds['ssh_exact'][i, :],
                                           levels=clevels, cmap=cmap,
                                           vmin=sol_min, vmax=sol_max,
                                           extend='both')
            ax[ncols - 1].set_aspect('equal', 'box')
            ax[ncols - 1].scatter(points[:, 0] / 1000,
                                  points[:, 1] / 1000, 15, 'k')

            ax[ncols - 1].set_title('Analytical solution')
            ax[ncols - 1].set_xlabel('x (km)')
            ax[ncols - 1].set_ylabel('y (km)')
            ds.close()

            cb = fig.colorbar(cm, ax=ax[-1], shrink=0.6)
            cb.set_label('ssh (m)')
            t = round((time[i] - time[0]).total_seconds() / 86400., 2)
            fig.suptitle(f'ssh solution at t={t} days')
            fig.savefig(f'solution_{i:03d}.png')
            plt.close()

    def rmse_plots(self):
        """
        Plot convergence curves
        """

        ramp_name = 'ramp'
        noramp_name = 'noramp'
        if self.use_lts:
            ramp_name = 'ramp_lts'
            noramp_name = 'noramp_lts'

        comparisons = []
        cases = {'standard_ramp': f'../../../standard/{ramp_name}/viz',
                 'standard_noramp': f'../../../standard/{noramp_name}/viz'}
        for case in cases:
            include = True
            for res in self.resolutions:
                if not os.path.exists(f'{cases[case]}/output_{res}km.nc'):
                    include = False
            if include:
                comparisons.append(case)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        max_rmse = 0
        for j, comp in enumerate(comparisons):
            rmse = np.zeros(len(self.resolutions))
            for i, res in enumerate(self.resolutions):

                rmse[i] = self.compute_rmse(
                    'h',
                    f'{cases[comp]}/output_{res}km.nc')

                if rmse[i] > max_rmse:
                    max_rmse = rmse[i]

            ax.loglog(self.resolutions, rmse,
                      linestyle='-', marker='o', label=comp)

        rmse_1st_order = np.zeros(len(self.resolutions))
        rmse_1st_order[0] = max_rmse
        for i in range(len(self.resolutions) - 1):
            rmse_1st_order[i + 1] = rmse_1st_order[i] / 2.0

        ax.loglog(self.resolutions, np.flip(rmse_1st_order),
                  linestyle='-', color='k', alpha=.25, label='1st order')

        ax.set_xlabel('Cell size (km)')
        ax.set_ylabel('RMSE (m)')

        ax.legend(loc='lower right')
        ax.set_title('Layer thickness convergence')
        fig.tight_layout()
        fig.savefig('error.png')

    def compute_rmse(self, varname, filename):
        """
        Compute the rmse between the modeled and exact solutions
        """

        ds = xr.open_dataset(filename)

        time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                for x in ds.xtime.values]
        ind = time.index(dt.datetime.strptime('0001-01-03_18:00:00',
                                              '%Y-%m-%d_%H:%M:%S'))
        if varname == 'zeta':
            var = ds['ssh'].values[ind, :]
        elif varname == 'h':
            var = ds['layerThickness'].values[ind, :, 0]

        t = (time[ind] - time[0]).total_seconds()
        var_exact = self.exact_solution(varname, ds.xCell.values,
                                        ds.yCell.values, t)
        rmse = np.sqrt(np.mean(np.square(var - var_exact)))

        return rmse

    def exact_solution(self, var, x, y, t):
        """
        Evaluate the exact solution
        """

        config = self.config

        f = config.getfloat('parabolic_bowl', 'coriolis_parameter')
        eta0 = config.getfloat('parabolic_bowl', 'eta_max')
        b0 = config.getfloat('parabolic_bowl', 'depth_max')
        omega = config.getfloat('parabolic_bowl', 'omega')
        g = config.getfloat('parabolic_bowl', 'gravity')

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        t = np.atleast_1d(t)

        if t.size > 1:
            x = np.resize(x, t.shape)
            y = np.resize(y, t.shape)

        eps = 1.0e-12
        r = np.sqrt(np.square(x) + np.square(y))
        L = np.sqrt(8.0 * g * b0 / (omega**2 - f**2))
        C = ((b0 + eta0)**2 - b0**2) / ((b0 + eta0)**2 + b0**2)
        b = b0 * (1.0 - r**2 / L**2)
        num = 1.0 - C**2
        den = 1.0 / (1.0 - C * np.cos(omega * t))
        h = b0 * (den * np.sqrt(num) - den**2 * (r**2 / L**2) * num)
        h[h < eps] = 0.0

        if var == 'h':
            soln = h

        elif var == 'zeta':
            soln = b0 * (den * np.sqrt(num) - 1.0 -
                         (r**2 / L**2) * (den**2 * num - 1.0))
            soln[h < eps] = -b[h < eps]

        elif var == 'u':
            soln = 0.5 * den * (omega * x * C * np.sin(omega * t) -
                                f * y * (np.sqrt(num) +
                                C * np.cos(omega * t) - 1.0))
            soln[h < eps] = 0

        elif var == 'v':
            soln = 0.5 * den * (omega * y * C * np.sin(omega * t) +
                                f * x * (np.sqrt(num) +
                                C * np.cos(omega * t) - 1.0))
            soln[h < eps] = 0

        elif var == 'r':
            soln = L * np.sqrt((1.0 - C * np.cos(omega * t)) /
                               np.sqrt(1.0 - C**2))

        else:
            print('Variable name not supported')

        return soln
