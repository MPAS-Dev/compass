import datetime as dt
import os
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mosaic
import numpy as np
import xarray as xr
from inpoly import inpoly2
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

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

        self.add_input_file(
            filename='buttermilk_bathy.nc',
            target='buttermilk_bathy.nc',
            database='buttermilk_bay')

        for res in resolutions:
            self.add_input_file(filename=f'output_{res}m.nc',
                                target=f'../forward_{res}m/output.nc')

    def run(self):
        """
        Run this step of the test case
        """

        self.resolutions = self.config.getlist('buttermilk_bay',
                                               'resolutions', dtype=int)

        plot_mode = 'paper'
        if plot_mode == 'paper':
            mpl.rcParams['mathtext.fontset'] = 'stix'
            mpl.rcParams['font.family'] = 'STIXGeneral'

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

        plot_mode = 'paper'
        fig, ax = plt.subplots(nrows=len(points), ncols=1,
                               figsize=(5, 1.5 * len(points)))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

        for j, res in enumerate(self.resolutions):
            filename = f'output_{res}m.nc'
            exists = self.check_file_exists(filename)
            if not exists:
                continue

            ds = xr.open_dataset(filename)

            time = [dt.datetime.strptime(x.decode(), '%Y-%m-%d_%H:%M:%S')
                    for x in ds.xtime.values]
            t = np.asarray([(x - time[0]).total_seconds() for x in time])

            xy = np.vstack((ds.xCell.values, ds.yCell.values)).T
            interp = NearestNDInterpolator(xy, ds.ssh.values.T)

            for i, pt in enumerate(points):

                ssh = interp(pt).T
                if res == 8:
                    label = 'reference'
                    color = 'k'
                else:
                    label = f'{res}m'
                    color = colors[j]
                ax[i].plot(t / 86400, ssh, label=label, color=color)

        for i, pt in enumerate(points):
            ax[i].set_xlabel('t (days)')
            ax[i].set_ylabel('ssh (m)')

            if plot_mode == 'paper':
                title = f'Station {i + 1}'
            else:
                title = f'Point ({pt[0] / 1000}, {pt[1] / 1000})'
            ax[i].set_title(title)
            if i == len(points) - 1:
                lines, labels = ax[i].get_legend_handles_labels()
            # ax[i].set_ylim(-2.5,2.5)

        titles = {'subgrid': 'a)', 'standard': 'b)'}
        if plot_mode == 'paper':
            title = titles[self.wetdry]
            ha = 'left'
            x = 0.0
            y = 0.98
        else:
            title = f'Buttermilk Bay ({self.wetdry})'
            ha = 'center'
            x = 0.5
            y = 0.98

        fig.suptitle(title, x=x, y=y, ha=ha, fontsize='x-large')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        fig.legend(lines, labels,
                   loc='lower center', ncol=3)
        fig.savefig('points.png', dpi=400)

    def contour_plots(self, points):
        """
        Plot contour plots at a specified output interval for each resolution
        and show where the points used in `points.png` are located.
        """

        sol_min = -2.0
        sol_max = 2.0
        cmap = plt.get_cmap('RdBu')

        minval = 0.2
        maxval = 0.8
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'truncated RdBu',
            cmap(np.linspace(minval, maxval, 256)))

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
                                   figsize=(3 * ncols, 3),
                                   constrained_layout=True)

            for j, res in enumerate(self.resolutions):
                filename = f'output_{res}m.nc'
                exists = self.check_file_exists(filename)
                if not exists:
                    continue
                ds = xr.open_dataset(filename)
                descriptor = mosaic.Descriptor(ds)
                cm = mosaic.polypcolor(ax[j], descriptor, ds['ssh'][i, :],
                                       vmin=sol_min, vmax=sol_max,
                                       cmap=cmap, antialiaseds=False)
                ax[j].set_aspect('equal', 'box')
                ax[j].set_title(f'{res}m resolution')
                ax[j].set_xlabel('x (km)')
                ax[j].set_ylabel('y (km)')
                ds.close()

                formatter = ticker.FuncFormatter(lambda x_val,
                                                 pos: f'{x_val / 1000:g}')

                ax[j].xaxis.set_major_formatter(formatter)
                ax[j].yaxis.set_major_formatter(formatter)

                ax[j].set_aspect('equal', 'box')
                ax[j].scatter(points[:, 0],
                              points[:, 1], 15, 'k')
                ax[j].set_xlim([0.0, 4000])
                ax[j].set_ylim([0.0, 3500])

            for j, sta in enumerate(range(len(points))):
                if (j == 2) | (j == 3):
                    xoffset = 50
                    yoffset = -200
                else:
                    xoffset = 0
                    yoffset = 100
                ax[0].text(points[j, 0] + xoffset,
                           points[j, 1] + yoffset, str(j + 1), color='k')

            tick_step = 0.5
            ticks = np.arange(sol_min, sol_max + tick_step, tick_step)
            cb = fig.colorbar(cm, ax=ax[-1], shrink=0.7,
                              ticks=ticks, extend='both')
            cb.set_label('ssh (m)')
            t = round((time[i] - time[0]).total_seconds() / 86400., 2)
            plot_mode = 'paper'
            if plot_mode == 'paper':
                titles = {'subgrid': 'a)', 'standard': 'c)'}
                title = titles[self.wetdry]
                ha = 'left'
                x = 0.0
                y = 0.98
            else:
                title = f'Buttermilk Bay ({self.wetdry}) ' \
                        f'ssh solution at t={t} days'
                ha = 'center'
                x = 0.5
                y = 0.98
            fig.suptitle(title, ha=ha, x=x, y=y, fontsize='x-large')
            fig.savefig(f'solution_{i:03d}.png', dpi=400)
            plt.close()

    def check_file_exists(self, path_str):
        p = Path(path_str)
        if p.exists():
            return True
        return False
