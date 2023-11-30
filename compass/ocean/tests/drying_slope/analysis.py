import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

from compass.step import Step


class Analysis(Step):
    """
    A step for analyzing the convergence of drying slope results and producing
    a convergence plot.

    Attributes
    ----------
    damping_coeff : float
        The Rayleigh damping coefficient used for the forward runs

    resolutions : float
        The resolution of the test case

    times : list of float
        The times at which to compare to the analytical solution
    """
    def __init__(self, test_case, resolutions, damping_coeff):
        """
        Create a forward step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolutions : list of floats
            The resolution of the test case

        damping_coeff: float
            the value of the rayleigh damping coefficient
        """

        super().__init__(test_case=test_case, name='analysis')

        self.damping_coeff = damping_coeff
        self.resolutions = resolutions
        self.times = ['0.05', '0.15', '0.25', '0.30', '0.40', '0.50']

        for resolution in resolutions:
            if resolution < 1.:
                res_name = f'{int(resolution*1e3)}m'
            else:
                res_name = f'{int(resolution)}km'
            self.add_input_file(filename=f'output_{res_name}.nc',
                                target=f'../forward_{res_name}/output.nc')
        for time in self.times:
            filename = f'r{damping_coeff}d{time}-analytical'\
                       '.csv'
            self.add_input_file(filename=filename, target=filename,
                                database='drying_slope')

        self.add_output_file(filename='convergence.png')

    def run(self):
        """
        Run this step of the test case
        """
        self._plot_convergence()

    def _compute_rmse(self, ds, t):
        """
        Get the time step

        Parameters
        ----------
        ds : xarray.Dataset
            the MPAS dataset containing output from a forward step
        t : float
            the time to evaluate the RMSE at

        Returns
        -------
        rmse : float
            the root-mean-squared-error of the MPAS output relative to the
            analytical solution at time t
        """

        x_exact, ssh_exact = self._exact_solution(t)
        tidx = int((float(t) / 0.2 + 1e-16) * 24.0)
        ds = ds.drop_vars(np.setdiff1d([j for j in ds.variables],
                                       ['yCell', 'ssh']))
        ds = ds.isel(Time=tidx)
        drying_length = self.config.getfloat('drying_slope', 'ly_analysis')
        drying_length = drying_length * 1e3
        x_offset = np.max(ds.yCell.values) - drying_length
        x_mpas = (ds.yCell.values - x_offset) / 1000.0
        ssh_mpas = ds.ssh.values
        # Interpolate mpas solution to the points at which we have an exact
        # solution
        idx_min = np.argwhere(x_exact - x_mpas[0] >= 0.).item(0)
        idx_max = np.argwhere(x_exact - x_mpas[-1] <= 0.).item(-1)
        f = interp1d(x_mpas, ssh_mpas)
        ssh_mpas_interp = f(x_exact[idx_min:idx_max])
        rmse = np.sqrt(np.mean(np.square(ssh_mpas_interp -
                                         ssh_exact[idx_min:idx_max])))
        return rmse

    def _plot_convergence(self):
        """
        Plot convergence curves
        """
        comparisons = []
        cases = {'standard': '../../../standard/convergence/analysis',
                 'ramp': '../../../ramp/convergence/analysis'}
        for case in cases:
            include = True
            for resolution in self.resolutions:
                if resolution < 1.:
                    res_name = f'{int(resolution*1e3)}m'
                else:
                    res_name = f'{int(resolution)}km'
                if not os.path.exists(f'{cases[case]}/output_{res_name}.nc'):
                    include = False
            if include:
                comparisons.append(case)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        max_rmse = 0
        for k, comp in enumerate(comparisons):
            rmse = np.zeros((len(self.resolutions), len(self.times)))
            for i, resolution in enumerate(self.resolutions):
                if resolution < 1.:
                    res_name = f'{int(resolution*1e3)}m'
                else:
                    res_name = f'{int(resolution)}km'
                ds = xr.open_dataset(
                    f'{cases[comp]}/output_{res_name}.nc')
                for j, time in enumerate(self.times):
                    rmse[i, j] = self._compute_rmse(ds, time)
            rmse_tav = np.mean(rmse, axis=1)
            if np.max(rmse_tav) > max_rmse:
                max_rmse = np.max(rmse_tav)
            ax.loglog(self.resolutions, rmse_tav,
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
        ax.set_title('SSH convergence')
        fig.tight_layout()
        fig.savefig('convergence.png')

    def _exact_solution(self, time):
        """
        Returns distance, ssh of the analytic solution
        """
        datafile = f'./r{self.damping_coeff}d{time}-'\
                   f'analytical.csv'
        data = pd.read_csv(datafile, header=None)
        return data[0], -data[1]
