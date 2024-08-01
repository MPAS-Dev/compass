import matplotlib.pyplot as plt
import numpy as np
import xarray

from compass.step import Step


class Viz(Step):
    """
    A step for visualizing output from baroclinic gyre

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case
        """
        super().__init__(test_case=test_case, name='viz')
        self.resolution = resolution

    def run(self):
        """
        Run this step of the test case
        """

        out_dir = '.'
        moc_dir = '../moc'
        mon_dir = '../forward/output'
        dsMesh = xarray.open_dataset('../initial_state/initial_state.nc')
        ds = xarray.open_dataset(f'{moc_dir}/moc.nc')
        # Insert plots here
        self._plot_moc(ds, dsMesh, out_dir)
        self._plot_spinup(mon_dir, dsMesh, out_dir)

    def _plot_moc(self, ds, dsMesh, out_dir):
        """
        Plot the time-mean moc state for the test case
        """
        avg_len = self.config.getint('mean_state_viz', 'time_averaging_length')
        moc = ds.moc[-12 * avg_len:, :, :].mean(axis=0).T.values
        latbins = ds.latBins
        plt.contourf(
            latbins, dsMesh.refInterfaces, moc,
            cmap="RdBu_r", vmin=-12, vmax=12)
        plt.gca().invert_yaxis()
        plt.ylabel('Depth (m)')
        plt.xlabel('Latitude')
        idx = np.unravel_index(np.argmax(moc), moc.shape)
        amoc = "max MOC = {:.1e}".format(round(np.max(moc), 1))
        maxloc = 'at lat = {} and z = {}m'.format(
            latbins[idx[-1]].values, int(dsMesh.refInterfaces[idx[0]].values))
        maxval = 'max MOC = {:.1e} at def loc'.format(
            round(np.max(moc[:, 175]), 1))
        plt.annotate(amoc + '\n' + maxloc + '\n' + maxval,
                     xy=(0.01, 0.05), xycoords='axes fraction')
        plt.colorbar()
        plt.savefig('{}/time_avg_moc_last{}years.png'.format(out_dir, avg_len))

    def _plot_spinup(self, mon_dir, dsMesh, out_dir):
        """
        Plot the timeseries of monthy layer-mean
        kinetic energy and temperature for the test case
        """

        ds = xarray.open_mfdataset(
            '{}/timeSeriesStatsMonthly*.nc'.format(mon_dir),
            concat_dim='Time', combine='nested')
        KE = ds.timeMonthly_avg_kineticEnergyCell[:, :, :].mean(axis=1)
        T = ds.timeMonthly_avg_activeTracers_temperature[:, :, :].mean(axis=1)
        midlayer = (dsMesh.refInterfaces +
                    0.5 *
                    (np.roll(dsMesh.refInterfaces, -1) - dsMesh.refInterfaces)
                    ).values[:-1]

        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        for ll in [0, 3, 6, 10, 14]:
            ax[0].plot(KE[:, ll], label='{}m'.format(int(midlayer[ll])))
            ax[1].plot(T[:, ll], label='{}m'.format(int(midlayer[ll])))
        ax[0].legend()
        ax[1].legend()
        ax[0].set_xlabel('Time (months)')
        ax[1].set_xlabel('Time (months)')
        ax[0].set_ylabel('Layer Mean Kinetic Energy ($m^2 s^{-2}$)')
        ax[1].set_ylabel(r'Layer Mean Temperature ($^{\circ}$C)')
        plt.savefig('{}/spinup_ft.png'.format(out_dir), bbox_inches='tight')
