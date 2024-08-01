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

        # config = self.config
        out_dir = '.'
        sim_dir = '../moc'
        dsMesh = xarray.open_dataset('../initial_state/initial_state.nc')
        ds = xarray.open_dataset(f'{sim_dir}/moc.nc')
        # Insert plots here
        self._plot_moc(ds, dsMesh, out_dir)

    def _plot_moc(self, ds, dsMesh, out_dir):
        """
        Plot the time-mean moc state for the test case
        """

        moc = ds.moc[:, :, :].mean(axis=0).T.values
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
        plt.savefig('{}/time_avg_moc.png'.format(out_dir))
