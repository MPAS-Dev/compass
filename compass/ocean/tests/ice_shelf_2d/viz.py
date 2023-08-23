import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray

from compass.ocean.tests.isomip_plus.viz.plot import MoviePlotter
from compass.step import Step


class Viz(Step):
    """
    A step for visualizing a cross-section through the 2D ice-shelf domain
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='viz')

        self.add_input_file(filename='initial_state.nc',
                            target='../initial_state/initial_state.nc')
        self.add_input_file(filename='output.nc',
                            target='../forward/output.nc')
        self.add_input_file(filename='output_ssh.nc',
                            target='../ssh_adjustment/output_ssh.nc')
        self.add_output_file('vert_grid.png')
        self.add_output_file('velocity_max_t.png')

    def run(self):
        """
        Run this step of the test case
        """

        ds = xarray.open_dataset('initial_state.nc')
        dsOut = xarray.open_dataset('output.nc')
        dsAdj = xarray.open_dataset('output_ssh.nc')
        plotter = MoviePlotter(inFolder='../forward',
                               streamfunctionFolder='',
                               outFolder='./plots',
                               expt='', sectionY=20. * 5.0e3,
                               dsMesh=ds, ds=dsOut,
                               showProgress=False)

        if 'Time' in ds.dims:
            ds = ds.isel(Time=0)

        ds = ds.groupby('yCell').mean(dim='nCells')

        nCells = ds.sizes['yCell']
        nVertLevels = ds.sizes['nVertLevels']

        zIndex = xarray.DataArray(data=np.arange(nVertLevels),
                                  dims='nVertLevels')

        minLevelCell = ds.minLevelCell - 1
        maxLevelCell = ds.maxLevelCell - 1

        cellMask = np.logical_and(zIndex >= minLevelCell,
                                  zIndex <= maxLevelCell)

        zIndex = xarray.DataArray(data=np.arange(nVertLevels + 1),
                                  dims='nVertLevelsP1')

        interfaceMask = np.logical_and(zIndex >= minLevelCell,
                                       zIndex <= maxLevelCell + 1)

        zMid = ds.zMid.where(cellMask)

        zInterface = np.zeros((nCells, nVertLevels + 1))
        zInterface[:, 0] = ds.ssh.values
        for zIndex in range(nVertLevels):
            thickness = ds.layerThickness.isel(nVertLevels=zIndex)
            thickness = thickness.fillna(0.)
            zInterface[:, zIndex + 1] = \
                zInterface[:, zIndex] - thickness.values

        zInterface = xarray.DataArray(data=zInterface,
                                      dims=('yCell', 'nVertLevelsP1'))
        zInterface = zInterface.where(interfaceMask)

        y = ds.yCell.values / 1.e3
        plt.figure(figsize=[12, 6], dpi=100)
        plt.plot(y, zMid.values, 'b', label='zMid')
        plt.plot(y, zInterface.values, 'k', label='zTop')
        plt.plot(y, ds.ssh.values, '--g', label='SSH')
        plt.plot(y, -ds.bottomDepth.values, '--r', label='zBed')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (m)')
        plt.legend()
        plt.savefig('vert_grid.png', dpi=200)
        plt.close()

        ds = xarray.open_dataset('initial_state.nc')

        # Plot the time series of max velocity
        plt.figure(figsize=[12, 6], dpi=100)
        umax = np.amax(dsOut.velocityX[:, :, 0].values, axis=1)
        vmax = np.amax(dsOut.velocityY[:, :, 0].values, axis=1)
        t = dsOut.daysSinceStartOfSim.values
        time = pd.to_timedelta(t) / 1.e9
        plt.plot(time, umax, 'k', label='u')
        plt.plot(time, vmax, 'b', label='v')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        plt.savefig('velocity_max_t.png', dpi=200)
        plt.close()
        min_column_thickness = self.config.getfloat(
            'ice_shelf_2d', 'y1_water_column_thickness')

        delssh = dsOut.ssh - dsOut.ssh[0, :]
        s_vmin = np.nanmin(delssh.values)
        s_vmax = np.nanmax(delssh.values)
        plotter.plot_horiz_series(delssh, 'delssh', 'delssh', True,
                                  cmap='cmo.curl',
                                  vmin=-1. * max(abs(s_vmin), abs(s_vmax)),
                                  vmax=max(abs(s_vmin), abs(s_vmax)))

        u_vmin = np.nanmin(dsOut.velocityX[:, :, 0].values)
        u_vmax = np.nanmax(dsOut.velocityX[:, :, 0].values)
        plotter.plot_horiz_series(dsOut.velocityX[:, :, 0], 'u', 'u', True,
                                  cmap='cmo.balance',
                                  vmin=-1. * max(abs(u_vmin), abs(u_vmax)),
                                  vmax=max(abs(u_vmin), abs(u_vmax)))

        v_vmin = np.nanmin(dsOut.velocityY[:, :, 0].values)
        v_vmax = np.nanmax(dsOut.velocityY[:, :, 0].values)
        plotter.plot_horiz_series(dsOut.velocityY[:, :, 0], 'v', 'v', True,
                                  cmap='cmo.balance',
                                  vmin=-1. * max(abs(v_vmin), abs(v_vmax)),
                                  vmax=max(abs(v_vmin), abs(v_vmax)))

        plotter.plot_horiz_series(dsOut.landIcePressure, 'landIcePressure',
                                  'landIcePressure',
                                  True, vmin=1e3,
                                  vmax=1e7, cmap_set_under='r',
                                  cmap_scale='log')

        plotter.plot_horiz_series(dsOut.ssh + ds.bottomDepth, 'H', 'H',
                                  True, vmin=min_column_thickness + 1e-10,
                                  vmax=2000, cmap_set_under='r',
                                  cmap_scale='log')

        delssh = dsAdj.ssh - ds.ssh
        s_vmin = np.nanmin(delssh.values)
        s_vmax = np.nanmax(delssh.values)
        plotter.plot_horiz_series(delssh, 'delssh_adjust', 'delssh_adjust',
                                  True, cmap='cmo.curl', time_indices=[0],
                                  vmin=-1. * max(abs(s_vmin), abs(s_vmax)),
                                  vmax=max(abs(s_vmin), abs(s_vmax)))
