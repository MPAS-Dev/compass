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

        self.add_input_file('../initial_state/initial_state.nc')
        self.add_input_file(
            '../forward/output/timeSeriesStatsMonthly.0001-01-01.nc')
        self.add_output_file('moc.nc')

    def run(self):
        """
        Run this step of the test case
        """

        in_dir = '../forward/output'
        out_dir = '.'

        lat_min = self.config.getfloat('baroclinic_gyre', 'lat_min')
        lat_max = self.config.getfloat('baroclinic_gyre', 'lat_max')
        dlat = self.config.getfloat('baroclinic_gyre_post', 'dlat')
        latBins = np.arange(lat_min + 2 * dlat,
                            lat_max + 2 * dlat, dlat)
        nz = self.config.getint('vertical_grid', 'vert_levels')

        dsMesh = xarray.open_dataset('../initial_state/initial_state.nc')

        ds = xarray.open_mfdataset(
            f'{in_dir}/timeSeriesStatsMonthly*.nc',
            concat_dim='Time', combine='nested')

        moc = self._compute_amoc(dsMesh, ds, latBins, nz)

        dsMOC = xarray.Dataset()
        dsMOC['xtime_startMonthly'] = ds.xtime_startMonthly
        dsMOC['xtime_endMonthly'] = ds.xtime_endMonthly
        dsMOC['moc'] = (["Time", "latBins", "nVertLevelsP1"], moc)
        dsMOC.coords['latBins'] = latBins
        dsMOC.coords['nVertLevelsP1'] = np.arange(nz + 1)
        dsMOC.moc.attrs['units'] = 'Sv'
        dsMOC.moc.attrs['description'] = \
            'zonally-averaged meridional overturning streamfunction'

        outputFileName = f'{out_dir}/moc.nc'
        write_netcdf(dsMOC, outputFileName)

    def _compute_amoc(self, dsMesh, ds, latBins, nz):
        """
        compute the overturning streamfunction for the given mesh
        """

        latCell = 180. / np.pi * dsMesh.variables['latCell'][:]
        nt = np.shape(ds.timeMonthly_avg_vertVelocityTop)[0]
        mocTop = np.zeros([nt, np.size(latBins), nz + 1])
        indlat_all = [np.logical_and(
            latCell >= latBins[iLat - 1], latCell < latBins[iLat])
            for iLat in range(1, np.size(latBins))]
        for tt in range(nt):
            for iLat in range(1, np.size(latBins)):
                indlat = indlat_all[iLat - 1]
                velArea = (ds.timeMonthly_avg_vertVelocityTop[tt, :, :] *
                           dsMesh.areaCell[:])
                mocTop[tt, iLat, :] = (mocTop[tt, iLat - 1, :] +
                                       np.nansum(velArea[indlat, :], axis=0))
        # convert m^3/s to Sverdrup
        mocTop = mocTop * 1e-6
        return mocTop
