import os

import numpy as np
import xarray
from mpas_tools.io import write_netcdf

from compass.step import Step


class Moc(Step):
    """
    A step for computing the zonally-averaged meridional overturning
    streamfunction in the single basin of the baroclinic gyre

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
        super().__init__(test_case=test_case, name='moc')
        self.resolution = resolution

        self.add_input_file('./init.nc')
        self.add_input_file(
            '../output/timeSeriesStatsMonthly.0001-01-01.nc')
        self.add_output_file('moc.nc')

    def run(self):
        """
        Run this step of the test case
        """

        in_dir = '../output'
        out_dir = '.'

        # show progress only if we're not writing to a log file
        # show_progress = self.log_filename is None

        lat_min = self.config.getfloat('baroclinic_gyre', 'lat_min')
        lat_max = self.config.getfloat('baroclinic_gyre', 'lat_max')
        dlat = 0.25  # set in config next
        # latbins = np.arange(15.5, 75.5, 0.25)
        latBins = np.arange(lat_min + dlat, lat_max + dlat, dlat)
        nz = self.config.getfloat('vertical_grid', 'vert_levels')

        dsMesh = xarray.open_dataset(os.path.join(in_dir, 'init.nc'))

        ds = xarray.open_mfdataset(
            '{}/timeSeriesStatsMonthly*.nc'.format(in_dir),
            concat_dim='Time', combine='nested')

        moc = self._compute_amoc(dsMesh, ds, latBins, nz)

        dsMOC = xarray.Dataset()
        dsMOC['xtime_startMonthly'] = ds.xtime_startMonthly
        dsMOC['xtime_endMonthly'] = ds.xtime_endMonthly
        dsMOC['moc'] = moc
        dsMOC.moc.attrs['units'] = 'Sv'
        dsMOC.moc.attrs['description'] = \
            'zonally-averaged meridional overturning streamfunction'
        outputFileName = '{}/moc.nc'.format(out_dir)
        # if file_complete(ds, outputFileName):
        #    return
        write_netcdf(dsMOC, outputFileName)

    def _compute_amoc(dsMesh, ds, latBins, nz):
        """
        compute the overturning streamfunction for the given mesh
        """

        latCell = 180. / np.pi * dsMesh.variables['latCell'][:]
        velArea = ds.vertVelocityTop.mean(axis=0) * ds.areaCell[:]
        mocTop = np.zeros([np.size(latBins), nz + 1])
        for iLat in range(1, np.size(latBins)):
            indlat = np.logical_and(
                latCell >= latBins[iLat - 1], latCell < latBins[iLat])
            mocTop[iLat, :] = mocTop[iLat - 1, :] \
                + np.nansum(velArea[indlat, :], axis=0)
        # convert m^3/s to Sverdrup
        mocTop = mocTop * 1e-6
        mocTop = mocTop.T
        return mocTop
