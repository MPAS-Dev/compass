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
    A step for visualizing buttermilk bay results

    Attributes
    ----------
    wetdry : str
        The wetting and drying approach used

     resolutions : list
         The grid resolutions run for this case
    """
    def __init__(self, test_case, wetdry, resolutions):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        wetdry : str
            The wetting and drying approach used

         resolutions : list
             The grid resolutions run for this case
        """
        super().__init__(test_case=test_case, name='viz')

        self.resolutions = resolutions
        self.wetdry = wetdry

        for res in resolutions:
            self.add_input_file(filename=f'output_{res}m.nc',
                                target=f'../forward_{res}m/output.nc')

    def run(self):
        """
        Run this step of the test case
        """

        points = self.get_points()
        self.timeseries_plots(points)
        self.contour_plots(points)

    def get_points(self):
        """
        Get the point coordinates for plotting solution timeseries
        """

        points = self.config.get('buttermilk_bay_viz', 'points')
        points = points.replace('[', '').replace(']', '').split(',')
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        points = points * 1000

        return points

    def timeseries_plots(self, points):
        """
        Plot solution timeseries at a given number of points
        for each resolution
        """

        fig, ax = plt.subplots(nrows=len(points), ncols=1,
                               figsize=(6, 2 * len(points)))

        for res in self.resolutions:
            ds = xr.open_dataset(f'output_{res}m.nc')

            time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                    for x in ds.xtime.values]
            t = np.asarray([(x - time[0]).total_seconds() for x in time])

            xy = np.vstack((ds.xCell.values, ds.yCell.values)).T
            interp = LinearNDInterpolator(xy, ds.ssh.values.T)

            for i, pt in enumerate(points):

                ssh = interp(pt).T
                ax[i].plot(t / 86400, ssh, label=f'{res}m')

        for i, pt in enumerate(points):
            ax[i].set_xlabel('t (days)')
            ax[i].set_ylabel('ssh (m)')
            ax[i].set_title(f'Point ({pt[0]/1000}, {pt[1]/1000})')
            if i == len(points) - 1:
                lines, labels = ax[i].get_legend_handles_labels()

        fig.suptitle(f'Buttermilk Bay ({self.wetdry})')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.legend(lines, labels,
                   loc='lower center', ncol=4)
        fig.savefig('points.png')

    def contour_plots(self, points):
        """
        Plot contour plots at a specified output interval for each resolution
        and show where the points used in `points.png` are located.
        """

        sol_min = -2
        sol_max = 2
        clevels = np.linspace(sol_min, sol_max, 50)
        cmap = plt.get_cmap('RdBu')

        ds = xr.open_dataset(f'output_{self.resolutions[0]}m.nc')
        time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                for x in ds.xtime.values]
        ds.close()

        plot_interval = self.config.getint('buttermilk_bay_viz',
                                           'plot_interval')
        for i, tstep in enumerate(time):

            if i % plot_interval != 0:
                continue

            ncols = len(self.resolutions)
            fig, ax = plt.subplots(nrows=1, ncols=ncols,
                                   figsize=(5 * ncols, 5),
                                   constrained_layout=True)

            for j, res in enumerate(self.resolutions):
                ds = xr.open_dataset(f'output_{res}m.nc')
                cm = ax[j].tricontourf(ds.xCell / 1000, ds.yCell / 1000,
                                       ds['ssh'][i, :],
                                       levels=clevels, cmap=cmap,
                                       vmin=sol_min, vmax=sol_max,
                                       extend='both')
                ax[j].set_aspect('equal', 'box')
                ax[j].set_title(f'{res}m resolution')
                ax[j].set_xlabel('x (km)')
                ax[j].set_ylabel('y (km)')
                ds.close()

                ax[j].set_aspect('equal', 'box')
                ax[j].scatter(points[:, 0] / 1000,
                              points[:, 1] / 1000, 15, 'k')

            cb = fig.colorbar(cm, ax=ax[-1], shrink=0.6)
            cb.set_label('ssh (m)')
            t = round((time[i] - time[0]).total_seconds() / 86400., 2)
            fig.suptitle((f'Buttermilk Bay ({self.wetdry}) '
                          f'ssh solution at t={t} days'))
            fig.savefig(f'solution_{i:03d}.png')
            plt.close()
